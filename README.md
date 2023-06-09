# NLP from Dostoyevsky books

## Backend
- Initial setup:
```bash
cd backend
pipenv shell
pipenv install --dev
```

- Makefile help:
```bash
make
```

- To execute tests:
```bash
pytest
```

- To build and run docker image with FastAPI app:
```bash
make dev-build-image tag=<imagetag>
make dev-run tag=<imagetag>
```
Set up the `imageatag` of your preference, for example: `fyodor:v1`. The app will be available at http://localhost:8000