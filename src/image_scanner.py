import cv2
import numpy as np
import threading
import queue

class ImageScanner:
    def __init__(self, input_dir, output_dir):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        
        # State Variables
        self.current_idx = 0
        self.image = None
        self.display_image = None
        self.points = []
        self.scale = 1.0     # 1.0 means fit scale. View width = Image width / scale
        self.offset_x = 0.5  # 0.0 to 1.0 (relative to image width)
        self.offset_y = 0.5  # 0.0 to 1.0 (relative to image height)
        self.window_name = "Scanner"
        self.cmd_queue = queue.Queue()
        self.running = True
        self.needs_redraw = True
        
        # Locks
        self.state_lock = threading.Lock()

    def load_images(self):
        self.image_files = sorted([
            f for f in self.input_dir.iterdir() 
            if f.is_file() and f.suffix.lower() in self.valid_extensions
        ])
        return len(self.image_files)

    def order_points(self, pts):
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        return rect

    def four_point_transform(self, image, pts):
        rect = self.order_points(pts)
        (tl, tr, br, bl) = rect
        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        maxWidth = max(int(widthA), int(widthB))
        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        maxHeight = max(int(heightA), int(heightB))
        dst = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]], dtype="float32")
        M = cv2.getPerspectiveTransform(rect, dst)
        return cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    def get_view_rect(self):
        """Calculate the source rectangle in the original image to display."""
        if self.image is None:
            return (0, 0, 100, 100)
            
        h, w = self.image.shape[:2]
        
        # Determine view size based on scale
        # Scale 1.0: Full image fits in view logic (though we might resize for display)
        # Scale > 1.0: Zoomed in (view is smaller portion of image)
        view_w = w / self.scale
        view_h = h / self.scale
        
        # Center constraint
        cx = self.offset_x * w
        cy = self.offset_y * h
        
        x1 = int(cx - view_w / 2)
        y1 = int(cy - view_h / 2)
        
        # Clamp to bounds
        # We allow Panning a bit freely but prevent going completely off
        if x1 < 0: x1 = 0
        if y1 < 0: y1 = 0
        if x1 + view_w > w: x1 = int(w - view_w)
        if y1 + view_h > h: y1 = int(h - view_h)
        # Double check
        if x1 < 0: x1 = 0
        if y1 < 0: y1 = 0
        
        return (x1, y1, int(view_w), int(view_h))

    def update_display(self):
        if self.image is None:
            return

        x, y, w, h = self.get_view_rect()
        
        # Crop from original
        ih, iw = self.image.shape[:2]
        x2 = min(x + w, iw)
        y2 = min(y + h, ih)
        x = max(0, x)
        y = max(0, y)
        
        try:
            crop = self.image[y:y2, x:x2]
        except:
            crop = self.image
        
        if crop.size == 0:
            return

        # Resize for display
        # We aim for a target display height, e.g. 800px, to keep UI responsive
        # The window size can be changed by user (WINDOW_NORMAL), but we feed it a fixed resolution image
        # for consistency. If user maximizes window, OpenCV scales this image up.
        render_h = 800
        aspect = crop.shape[1] / crop.shape[0] if crop.shape[0] > 0 else 1
        render_w = int(render_h * aspect)
        
        self.display_image = cv2.resize(crop, (render_w, render_h), interpolation=cv2.INTER_LINEAR)
        
        # Draw points
        scale_x = render_w / w if w > 0 else 1
        scale_y = render_h / h if h > 0 else 1
        
        for idx, pt in enumerate(self.points):
            px, py = pt
            # Check if point is inside current view
            if x <= px < x + w and y <= py < y + h:
                dx = int((px - x) * scale_x)
                dy = int((py - y) * scale_y)
                cv2.circle(self.display_image, (dx, dy), 5, (0, 0, 255), -1)
                cv2.putText(self.display_image, str(idx+1), (dx+10, dy+10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Add Overlay Info
        info = f"Zoom: {self.scale:.2f}x | Points: {len(self.points)}/4"
        cv2.putText(self.display_image, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        cv2.imshow(self.window_name, self.display_image)
        self.needs_redraw = False

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            with self.state_lock:
                if self.display_image is None: return
                
                dh, dw = self.display_image.shape[:2]
                vx, vy, vw, vh = self.get_view_rect()
                
                # Map display coord (x,y) -> View coord -> Original coord
                rx = x / dw
                ry = y / dh
                
                ox = vx + rx * vw
                oy = vy + ry * vh
                
                self.points.append((ox, oy))
                print(f"已记录点: ({int(ox)}, {int(oy)})")
                self.needs_redraw = True

    def process_command(self, cmd):
        # Handle command
        cmd = cmd.strip().lower()
        if not cmd: return
        
        with self.state_lock:
            # Zoom
            if any(c in cmd for c in ['zoom +', 'zoom in', 'z+', '放大']):
                self.scale *= 1.2
            elif any(c in cmd for c in ['zoom -', 'zoom out', 'z-', '缩小']):
                self.scale /= 1.2
            
            # Move
            elif any(c in cmd for c in ['up', '向上']):
                self.offset_y -= 0.1 / self.scale
            elif any(c in cmd for c in ['down', '向下']):
                self.offset_y += 0.1 / self.scale
            elif any(c in cmd for c in ['left', '向左']):
                self.offset_x -= 0.1 / self.scale
            elif any(c in cmd for c in ['right', '向右']):
                self.offset_x += 0.1 / self.scale
                
            # Undo
            elif any(c in cmd for c in ['undo', 'back', '撤回']):
                if self.points:
                    self.points.pop()
                    print("已撤回上一个点")
            
            # Reset
            elif cmd in ['reset', 'r']:
                self.scale = 1.0
                self.offset_x = 0.5
                self.offset_y = 0.5
            
            # Help
            elif cmd in ['h', 'help', '?']:
                print("\n=== 指令列表 ===")
                print("缩放: zoom +, zoom -, 放大, 缩小")
                print("移动: up, down, left, right (或 w/a/s/d 键)")
                print("操作: undo (撤回), reset (重置视图), next (跳过), quit (退出)")
            
            # Clamp values
            self.scale = max(0.1, min(self.scale, 50.0))
            self.offset_x = max(0.0, min(1.0, self.offset_x))
            self.offset_y = max(0.0, min(1.0, self.offset_y))
            
            self.needs_redraw = True

    def input_thread_func(self):
        print(">> 终端指令模式就绪 (输入 help 查看常用指令)")
        while self.running:
            try:
                # Use sys.stdin to avoid some issues, though input() is usually fine
                cmd = input()
                if cmd.strip():
                    self.cmd_queue.put(cmd)
            except EOFError:
                break
            except Exception:
                pass

    def run(self):
        if self.load_images() == 0:
            print("未找到图片。")
            return

        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)
        
        # Start Input Thread
        t = threading.Thread(target=self.input_thread_func, daemon=True)
        t.start()
        
        print(f"开始处理 {len(self.image_files)} 张图片...")
        
        while self.current_idx < len(self.image_files):
            file = self.image_files[self.current_idx]
            print(f"\n[{self.current_idx+1}/{len(self.image_files)}] 载入: {file.name}")
            
            self.image = cv2.imread(str(file))
            if self.image is None:
                print("读取失败，跳过")
                self.current_idx += 1
                continue
                
            # Reset view
            with self.state_lock:
                self.points = []
                self.scale = 1.0 # Reset zoom
                self.offset_x = 0.5
                self.offset_y = 0.5
                self.needs_redraw = True
            
            image_completed = False
            while not image_completed and self.running:
                # Process Queue
                try:
                    while True:
                        cmd = self.cmd_queue.get_nowait()
                        if cmd in ['quit', 'exit', 'q']:
                            self.running = False
                            return
                        elif cmd == 'next':
                            image_completed = True # Skip current
                        else:
                            self.process_command(cmd)
                except queue.Empty:
                    pass

                # Check completion
                with self.state_lock:
                    if len(self.points) == 4:
                        print("检测到4个点，正在裁剪...")
                        try:
                            pts = np.array(self.points, dtype="float32")
                            warped = self.four_point_transform(self.image, pts)
                            out_path = self.output_dir / file.name
                            cv2.imwrite(str(out_path), warped)
                            print(f"保存成功: {out_path.name}")
                            image_completed = True
                        except Exception as e:
                            print(f"裁剪失败: {e}")
                            self.points = []
                            self.needs_redraw = True

                if self.needs_redraw:
                    self.update_display()

                # Key handling
                key = cv2.waitKey(50) & 0xFF
                if key != 255:
                    if key == ord('q'): # Quit
                        self.running = False
                        return
                    elif key == ord('='): self.process_command('zoom +')
                    elif key == ord('-'): self.process_command('zoom -')
                    elif key == ord('w'): self.process_command('up')
                    elif key == ord('s'): self.process_command('down')
                    elif key == ord('a'): self.process_command('left')
                    elif key == ord('d'): self.process_command('right')
                    elif key == ord('z'): self.process_command('undo')
                    elif key == ord('r'): self.process_command('reset')
                    elif key == ord('n'): self.cmd_queue.put('next')

            if not self.running: break
            self.current_idx += 1
            
        cv2.destroyAllWindows()
        print("所有图片处理完毕。")
