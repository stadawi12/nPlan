all: 
	$(MAKE) testall || $(MAKE) clean

testall:
	rm runs/* -rf
	rm models/*.pt -f
	python test_data.py
	python test_loader.py
	python test_models.py
	rm runs/* -r
	rm models/*.pt

clean:
	rm runs/* -r
	rm models/*.pt
