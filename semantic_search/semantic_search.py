from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.utils import embedding_functions


# Example documents (in practice, load your data from files or database)
documents = [
    "A dog is a domesticated carnivore of the family Canidae.",        # info about a dog
    "A cat is a small domesticated carnivorous mammal.",              # info about a cat
    "The stock market is where shares in companies are bought and sold.",  # info about stock market
    "Apple pie is a popular dessert made from apples and pastry.",    # info about apple pie
    "The kitten purrs softly while playing with a toy."              # info about a kitten (young cat)
]


# Load a pre-trained embedding model
# a popular lightweight model (384-dim embeddings)
model = SentenceTransformer('all-MiniLM-L6-v2')

# Compute embeddings for all documents
doc_embeddings = model.encode(documents)  # this returns a list/array of vector embeddings
print(f"Generated {len(doc_embeddings)} embeddings of dimension {len(doc_embeddings[0])}.")

# Initialize a Chroma client (in-memory for demo; could specify persistence directory)
client = chromadb.Client()

# Create a collection in the vector database
collection = client.create_collection(name="doc_search")


# Add documents to the collection with their embeddings
# We provide the text (documents), unique ids, and the pre-computed embeddings.
collection.add(
    documents=documents,
    embeddings=doc_embeddings.tolist(),  # ensure it's a list of lists (if it's a numpy array, convert to list)
    ids=[f"doc{i}" for i in range(len(documents))]
)
print("Documents added to vector database. Number of items in collection:", collection.count())

# User query
query = "Sweet Fruit Pastry"
# Embed the query to a vector
query_embedding = model.encode([query])  # encode expects a list, we give [query]

# Perform similarity search in the collection
results = collection.query(
    query_embeddings=query_embedding,
    n_results=3,  # find the top-3 nearest neighbors
    include=["documents", "distances"]  # also return the actual text and distance scores
)
for doc, distance in zip(results["documents"][0], results["distances"][0]):
    print(f"Result: {doc} (distance: {distance:.4f})")
