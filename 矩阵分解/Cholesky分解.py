import numpy as np

def cholesky_decomposition(A):
    """
    对Hermite正定矩阵A进行Cholesky分解。
    参数:
    A : numpy.ndarray
        输入的Hermite正定矩阵

    返回:
    G : numpy.ndarray
        下三角矩阵G，使得 A = G * G.T
    """
    n = A.shape[0]
    G = np.zeros_like(A)

    for i in range(n):
        # 对角线元素 g_ii
        sum_g2 = sum(G[i, k] ** 2 for k in range(i))
        G[i, i] = np.sqrt(A[i, i] - sum_g2)
        
        # 非对角线元素 g_ij
        for j in range(i + 1, n):
            sum_gk = sum(G[i, k] * G[j, k] for k in range(i))
            G[j, i] = (A[j, i] - sum_gk) / G[i, i]
            
    return G

# 示例矩阵
A = np.array([
    [5, 2, -4],
    [2, 1, -2],
    [-4, -2, 5]
], dtype=float)

# 调用Cholesky分解函数
G = cholesky_decomposition(A)
print("Cholesky分解得到的下三角矩阵G为：")
print(G)
print("验证计算机结果：G * G.T = ")
print(np.dot(G, G.T))
print("计算机结果与原矩阵误差：")
print(A-np.dot(G, G.T))


# 手算结果验证
Hand = np.array([
    [np.sqrt(5), 0, 0],
    [2*np.sqrt(5)/5, np.sqrt(5)/5, 0],
    [-4*np.sqrt(5)/5, -2*np.sqrt(5)/5, 1]
], dtype=float)
print("手算的下三角矩阵G为：")
print(Hand)
print("验证手算结果：G*G.T=")
print(np.dot(Hand, Hand.T))
print("手算结果与原矩阵误差：")
print(A-np.dot(Hand, Hand.T))

# 算法与手算精度差
print("算法与手算精度差：")
print(np.dot(G, G.T)-np.dot(Hand, Hand.T))