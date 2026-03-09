default: run

clean:
	rm -rf ./input/*
	rm -rf ./output/*

run:
	uv run main.py