# Challenge 1B: Persona-Driven Document Intelligence

## Approach
This solution ranks document sections based on persona and job relevance using TF-IDF vectorization and cosine similarity.

### Key Features:
- Generic persona/job handling for any domain
- TF-IDF based semantic similarity scoring
- Top-10 section ranking system
- Fast CPU-only processing

### Algorithm:
1. Load JSON documents from Challenge 1A
2. Extract all sections with metadata
3. Combine persona + job into query vector
4. Calculate TF-IDF vectors for all sections
5. Compute cosine similarity scores
6. Rank and return top 10 sections

## Libraries Used
- **scikit-learn**: TF-IDF vectorization and cosine similarity
- **numpy**: Numerical operations

## How to Build and Run

### Build:
```bash
docker build --platform linux/amd64 -t doc-intelligence:v1 .
```
Run (Windows PowerShell):
```bash
docker run --rm -v ${PWD}/input:/app/input -v ${PWD}/output:/app/output --network none doc-intelligence:v1
```
Run (Windows CMD):
```bash
docker run --rm -v %cd%/input:/app/input -v %cd%/output:/app/output --network none doc-intelligence:v1
```
Run (Linux/macOS):
```bash
docker run --rm -v %cd%/input:/app/input -v %cd%/output:/app/output --network none doc-intelligence:v1
```
Performance
- Processes 3-10 documents in under 60 seconds
- Works offline, CPU-only
- Model size under 1GB

