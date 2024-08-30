import ordpy

def get_smoothness_structure(image, taux=1, tauy=1, q=0):
    '''
        Returns the smoothness and curve structure of given image
    '''
    ord_dis = ordpy.weighted_smoothness_structure(image, taux, tauy, q=q)
    print(f"Result: Smoothness = {ord_dis[0]}; Curve Structure = {ord_dis[1]};")
    
    return ord_dis[0], ord_dis[1]