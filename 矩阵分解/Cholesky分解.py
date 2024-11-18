import numpy as np

def cholesky_decomposition(A):
    """
    对 Hermitian 正定矩阵 A 进行 Cholesky 分解。
    支持复数矩阵。
    
    参数:
    A : numpy.ndarray
        输入的 Hermitian 正定矩阵

    返回:
    G : numpy.ndarray
        下三角矩阵 G，使得 A = G * G.H
    """
    # 检查矩阵是否Hermite
    if not np.allclose(A, A.conj().T):
        raise ValueError("输入矩阵不是Hermite矩阵")
    
    n = A.shape[0] # 矩阵A的阶数
    G = np.zeros_like(A, dtype=complex) # 初始化下三角阵

    for i in range(n):
        # 对角线元素 g_ii，先行计算
        sum_g2 = sum(G[i, k] * G[i, k].conj() for k in range(i)) # 计算g_i1,g_i2,...,g_i,i-1的平方和
        G[i, i] = np.sqrt(A[i, i] - sum_g2) # a_ii减去sum_g2即可得到g_ii值
        
        # 非对角线元素 g_ij，再列计算
        for j in range(i + 1, n):
            sum_gk = sum(G[i, k] * G[j, k].conj() for k in range(i)) # 计算g_i1*g_j1,...,g_{i,i-1}*g_{j,i-1}之和
            G[j, i] = (A[j, i] - sum_gk) / G[i, i].conj() # 减掉再除以即可
            
    return G # 输出下三角阵

if __name__ == "__main__":
    # 示例：复数矩阵
    A = np.array([
        [4+0j, 1-1j, 2+0j],
        [1+1j, 3+0j, 0-1j],
        [2+0j, 0+1j, 2+0j]
    ], dtype=complex)

    # 调用 Cholesky 分解函数
    G = cholesky_decomposition(A)
    print("Cholesky 分解得到的下三角矩阵 G 为：")
    print(G)

    # 验证分解结果
    print("验证结果：G * G.H =")
    print(np.dot(G, G.conj().T))  # 使用共轭转置验证结果
    print("原矩阵与分解结果误差：")
    print(A - np.dot(G, G.conj().T))

    # # 示例：实数矩阵
    # A = np.array([
    #     [5., 2., -4.],
    #     [2., 1., -2.],
    #     [-4., -2., 5.]
    # ], dtype=complex)
    # # 调用Cholesky分解函数
    # G = cholesky_decomposition(A)
    # print("Cholesky分解得到的下三角矩阵G为：")
    # print(G)
    # print("验证计算机结果：G * G.H = ")
    # print(np.dot(G, G.conj().T))
    # print("计算机结果与原矩阵误差：")
    # print(A-np.dot(G, G.conj().T))
    # # 手算结果验证
    # Hand = np.array([
    #     [np.sqrt(5), 0, 0],
    #     [2*np.sqrt(5)/5, np.sqrt(5)/5, 0],
    #     [-4*np.sqrt(5)/5, -2*np.sqrt(5)/5, 1]
    # ], dtype=complex)
    # print("手算的下三角矩阵Hand为：")
    # print(Hand)
    # print("验证手算结果：Hand*Hand.H=")
    # print(np.dot(Hand, Hand.conj().T))
    # print("手算结果与原矩阵误差：")
    # print(A-np.dot(Hand, Hand.conj().T))
    # # 算法与手算精度差
    # print("算法与手算精度差：")
    # print(np.dot(G, G.conj().T)-np.dot(Hand, Hand.conj().T))