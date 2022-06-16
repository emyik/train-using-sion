DATE = $(shell date "+%Y%m%d%H%M")

test:
	@mkdir -p data/$(DATE)
	python3 ./__main__.py --dataset cifar --batch 64 --model resnet --test_mode --epochs 10 -o ./data/$(DATE)/test.csv $(PARAMS) 1>./data/$(DATE).log 2>&1 &
	@echo "Data output to data/$(DATE)/test.csv"
	@echo "Run \"tail -f ./data/$(DATE).log\" to see the training progress"

train-cifar:
	@mkdir -p data/$(DATE)
	python3 ./__main__.py --dataset cifar --batch 64 --model resnet -o ./data/$(DATE)/cifar_resnet_buildin.csv $(PARAMS) 1>./data/$(DATE).log 2>&1 &
	@echo "Data output to data/$(DATE)/cifar_resnet_buildin.csv"
	@echo "Run \"tail -f ./data/$(DATE).log\" to see the training progress"