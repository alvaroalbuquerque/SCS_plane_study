import ordpy

def get_entropy_complexity(image, dx=3, dy=1, taux=1, tauy=1):
    '''
        Returns the permutation entropy and fisher-shannon complexity measure of given image
    '''
    h, c = ordpy.complexity_entropy(image, dx, dy, taux, tauy)
    # print(f"Result: Entropy = {h}; Complexity Measure = {c};")
    
    return h, c

def get_weighted_entropy_complexity(image, dx=3, dy=1, taux=1, tauy=1, q=2):
    '''
        Returns the permutation entropy and statistical complexity measure of given image
    '''
    h, c = ordpy.weighted_complexity_entropy(image, dx, dy, taux, tauy, q)
    # print(f"Result: Entropy = {h}; Complexity Measure = {c};")
    
    return h, c

def get_weighted_fisher_shannon(image, dx=3, dy=1, taux=1, tauy=1, q=2):
    '''
        Returns the normalized permutation entropy and fisher-shannon complexity measure of given image
    '''
    h, f = ordpy.weighted_fisher_shannon(image, dx, dy, taux, tauy, q)
    # print(f"Result: Entropy = {h}; Fisher-Shannon complexity = {f};")

    return h, f