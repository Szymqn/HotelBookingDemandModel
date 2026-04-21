#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = HotelBookingDemandModel
PYTHON_VERSION = 3.13
PYTHON_INTERPRETER = python

#################################################################################
# COMMANDS                                                                      #
#################################################################################


## Install Python dependencies
.PHONY: requirements
requirements:
	$(PYTHON_INTERPRETER) -m pip install -U pip
	$(PYTHON_INTERPRETER) -m pip install -r requirements.txt
	



## Delete all compiled Python files
.PHONY: clean
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete


## Lint using ruff (use `make format` to do formatting)
.PHONY: lint
lint:
	ruff format --check
	ruff check

## Format source code with ruff
.PHONY: format
format:
	ruff check --fix
	ruff format



## Run tests
.PHONY: test
test:
	python -m pytest tests


## Set up Python interpreter environment
.PHONY: create_environment
create_environment:
	
	conda create --name $(PROJECT_NAME) python=$(PYTHON_VERSION) -y
	
	@echo ">>> conda env created. Activate with:\nconda activate $(PROJECT_NAME)"
	



#################################################################################
# PROJECT RULES                                                                 #
#################################################################################

#################################################################################
# MODELING PIPELINE RULES													 	#
#################################################################################

## Make dataset
.PHONY: data
data: requirements
	$(PYTHON_INTERPRETER) bookwiseai/dataset.py

## Make features engineering
.PHONY: features
features: data
	$(PYTHON_INTERPRETER) bookwiseai/features.py

## Make features engineering only
.PHONY: features_only
features_only: requirements
	$(PYTHON_INTERPRETER) bookwiseai/features.py


## Make feature selection
.PHONY: feature_selection
feature_selection: features
	$(PYTHON_INTERPRETER) bookwiseai/feature_selection.py

## Make feature selection only
.PHONY: feature_selection_only
feature_selection_only: requirements
	$(PYTHON_INTERPRETER) bookwiseai/feature_selection.py

## Make training
.PHONY: train
train: feature_selection
	$(PYTHON_INTERPRETER) bookwiseai/modeling/train.py

## Make training only
.PHONY: train_only
train_only: requirements
	$(PYTHON_INTERPRETER) bookwiseai/modeling/train.py

## Make prediction
.PHONY: predict
predict: train
	$(PYTHON_INTERPRETER) bookwiseai/modeling/predict.py

## Make prediction only
.PHONY: predict_only
predict_only: requirements
	$(PYTHON_INTERPRETER) bookwiseai/modeling/predict.py

#################################################################################
# EXPLANATION PIPELINE RULES (After modeling)												#
#################################################################################

# Instance level

## Make break down explanation
.PHONY: break_down
break_down: requirements
	$(PYTHON_INTERPRETER) bookwiseai/explanation/instance_level/break_down.py

## Make Shapley explanation
.PHONY: shap
shap: requirements
	$(PYTHON_INTERPRETER) bookwiseai/explanation/instance_level/shap.py

## Make LIME explanation
.PHONY: lime
lime: requirements
	$(PYTHON_INTERPRETER) bookwiseai/explanation/instance_level/LIME.py

## Make Ceteris-paribus Profiles explanation
.PHONY: cp
cp: requirements
	$(PYTHON_INTERPRETER) bookwiseai/explanation/instance_level/cp.py

# Dataset Level

## Make variable importance measure explanation
.PHONY: vim
vim: requirements
	$(PYTHON_INTERPRETER) bookwiseai/explanation/dataset_level/vim.py

## Make partial dependence plot explanation
.PHONY: pdp
pdp: requirements
	$(PYTHON_INTERPRETER) bookwiseai/explanation/dataset_level/pdp.py

## Make local dependence and accumulated local explanation
.PHONY: ld_and_al
ld_and_al: requirements
	$(PYTHON_INTERPRETER) bookwiseai/explanation/dataset_level/ld_and_al.py

## Make residual diagnostics explanation
.PHONY: rd
rd: requirements
	$(PYTHON_INTERPRETER) bookwiseai/explanation/dataset_level/rd.py

#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

define PRINT_HELP_PYSCRIPT
import re, sys; \
lines = '\n'.join([line for line in sys.stdin]); \
matches = re.findall(r'\n## (.*)\n[\s\S]+?\n([a-zA-Z_-]+):', lines); \
print('Available rules:\n'); \
print('\n'.join(['{:25}{}'.format(*reversed(match)) for match in matches]))
endef
export PRINT_HELP_PYSCRIPT

help:
	@$(PYTHON_INTERPRETER) -c "${PRINT_HELP_PYSCRIPT}" < $(MAKEFILE_LIST)
