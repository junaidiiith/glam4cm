from re import finditer
import numpy as np


SEP = ' '
def camel_case_split(identifier):
    matches = finditer('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)', identifier)
    return [m.group(0) for m in matches]


def doc_tokenizer(doc):
    words = doc.split()
    # split _
    words = [w2 for w1 in words for w2 in w1.split('_') if w2 != '']
    # camelcase
    words = [w2.lower() for w1 in words for w2 in camel_case_split(w1) if w2 != '']
    return words



def oversample_dataset(dataset, oversampling_ratio=0.7):
    """
    This function oversamples the classes that occur less frequently in the dataset.
    The occurence of each class is counted and each class is oversampled 70% of the difference between the most common class and the class in question.
    """

    class_occurences = dataset[:]['labels'].numpy()
    unique_classes, counts = np.unique(class_occurences, return_counts=True)
    max_count = counts.max()
    indices_with_oversamples = []
    for class_idx, count in zip(unique_classes, counts):
        class_indices = np.where(class_occurences == class_idx)[0]
        indices_with_oversamples.extend(class_indices)
        oversample_count = int(oversampling_ratio * (max_count - count))
        indices_with_oversamples.extend(np.random.choice(class_indices, oversample_count))
    
    return indices_with_oversamples