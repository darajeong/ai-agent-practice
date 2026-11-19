from embedding_search import cosine_similarity

print(cosine_similarity([1, 0], [1, 0]))
print(cosine_similarity([1, 0], [0, 1]))
print(cosine_similarity([0, 0], [1, 0]))

