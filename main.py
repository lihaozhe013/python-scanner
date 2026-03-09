from pathlib import Path
import cv2
import numpy as np

base_dir = Path(__file__).parent.resolve()
input_dir = base_dir / 'input'
output_dir = base_dir / 'output'

# 用于存储鼠标点击的4个坐标点
points = []

def order_points(pts):
    """
    将4个点按照：左上、右上、右下、左下的顺序排列
    """
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)] # 左上角：x+y的和最小
    rect[2] = pts[np.argmax(s)] # 右下角：x+y的和最大
    
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)] # 右上角：y-x最小
    rect[3] = pts[np.argmax(diff)] # 左下角：y-x最大
    return rect

def four_point_transform(image, pts):
    """
    执行透视变换和裁剪
    """
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    # 计算新图片的宽度（取上下边缘较长的那个）
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    # 计算新图片的高度（取左右边缘较长的那个）
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    # 构建目标点坐标（完美铺满矩形）
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    # 获取透视变换矩阵并执行
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped

def mouse_click(event, x, y, flags, param):
    """
    处理鼠标点击事件
    """
    global points
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        print(f"已记录点: ({x}, {y})，当前共 {len(points)} 个点")
        # 在图片上画个红点反馈一下
        cv2.circle(clone, (x, y), 5, (0, 0, 255), -1)
        cv2.imshow("Image", clone)

# ================== 主程序 ==================
if __name__ == "__main__":
    # 确保输出目录存在
    output_dir.mkdir(parents=True, exist_ok=True)

    # 获取输入目录下的所有支持的图片文件
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    image_files = [
        f for f in input_dir.iterdir() 
        if f.is_file() and f.suffix.lower() in valid_extensions
    ]
    
    if not image_files:
        print(f"在 {input_dir} 中未找到图片。")
        exit()

    print(f"找到 {len(image_files)} 张图片，即将开始处理。")
    print("操作说明：在每张图片上依次点击文档的四个角（顺序不限）。点满4个后自动保存并切换下一张。")
    print("按 'c' 清空当前图片选点，按 'q' 退出程序。")

    # 创建窗口并绑定鼠标事件
    cv2.namedWindow("Image")
    cv2.setMouseCallback("Image", mouse_click)

    for i, image_file in enumerate(image_files):
        print(f"[{i+1}/{len(image_files)}] 正在处理: {image_file.name}")
        
        # 读取图片
        image_path = str(image_file)
        image = cv2.imread(image_path)
        
        if image is None:
            print(f"无法读取图片: {image_file.name}，跳过。")
            continue

        # 重置状态
        clone = image.copy()
        points = []

        # 循环等待用户点击
        while True:
            cv2.imshow("Image", clone)
            key = cv2.waitKey(1) & 0xFF
            
            # 如果点够了4个，自动执行裁剪
            if len(points) == 4:
                print("4个点已收集，正在处理...")
                try:
                    pts = np.array(points, dtype="float32")
                    warped = four_point_transform(image, pts)
                    
                    # 保存结果到 output 目录，文件名不变
                    output_path = output_dir / image_file.name
                    cv2.imwrite(str(output_path), warped)
                    print(f"处理完成！已保存为: {output_path.name}")
                except Exception as e:
                    print(f"处理图片 {image_file.name} 时出错: {e}")
                
                # 处理完当前图片，跳出 while 循环，进入下一张for循环
                break
            
            # 按 'c' 键清空重来
            if key == ord("c"):
                clone = image.copy()
                points = []
                print("已清空点，请重新点击。")
            
            # 按 'q' 键退出
            elif key == ord("q"):
                print("用户退出。")
                cv2.destroyAllWindows()
                exit()
    
    cv2.destroyAllWindows()
    print("所有图片处理完成，程序退出。")