numpy:
	python3 execution_perf.py numpy

cupy:
	python3 execution_perf.py cupy

pytorch:
	python3 execution_perf.py pytorch

algo:
	python3 execution_perf.py algo

res:
	python3 execution_perf res

time:
	python3 execution_perf time

comp:
	python3 compare_execution_perf.py

make comp_res:
	python3 compare_execution_perf.py res

make comp_time:
	python3 compare_execution_perf.py time

test:
	pytest Unit_Tests/test_cupy.py && pytest Unit_Tests/test_pytorch.py

requirements:
	pip install -r requirements.txt

.PHONY: docs

generate:
	sphinx-apidoc -o doc ./ && sphinx-apidoc -o doc Algorithmic_Optimize/ && sphinx-apidoc -o doc Cupy_Optimize/ && sphinx-apidoc -o doc Pytorch_Optimize/

docs: generate
	cd doc && make html
