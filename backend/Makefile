.PHONY: help
help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

.PHONY: dev-build-image
dev-build-image: ## Build the image
	docker build -t $(tag) .

.PHONY: dev-run
dev-run: ## Run the container
	docker run -it --rm -e PORT=8000 -p 8000:8000 -v $(shell pwd):/app/ $(tag)