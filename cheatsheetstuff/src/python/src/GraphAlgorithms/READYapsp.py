#all pairs shortest path
# works on allpairspath, kastenlauf(modified), arbitrage
def APSP():
    for k in range(n):
        for i in range(n):
            for j in range(n):
                mat[i][j]=min(mat[i][j], mat[i][k]+mat[k][j])
