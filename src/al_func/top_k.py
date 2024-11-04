#This function returns k points of their index that has the largest value 
#values_with_indeces is an array of n 2-tuple (n>=k), where each tuple consists of value and index
def top_k_indices(k, values_with_indices):
    top_k = sorted(values_with_indices, key=lambda x: x[0], reverse=True)[:k]
    return [index for value, index in top_k]

#k = 3
#values_with_indices = [(5.2, 0), (3.1, 1), (7.6, 2), (2.9, 3), (6.4, 4)]
#top_indices = top_k_indices(k, values_with_indices)
#print("Top k indices:", top_indices)
