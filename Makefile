init:
	@pip install -r requirements.txt # 安装Python第三方库
	@unzip data.zip # 解压缩出Kaggle提供的数据集和提交样例，都是csv格式

run:
	@python main.py

.PHONY: clean
clean:
	@rm *.csv
	@rm -rf __pycache__