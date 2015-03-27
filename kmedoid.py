
import copy, random, math

import numpy as np

class Cluster:
    def __init__(self, dm):
        self.clearMember()
        self.distance_matrix = dm

    def setCenter(self, c):
        self.center = c
        self.addMember(c)

    def addMember(self, m):
        self.members.append(m)

    def clearMember(self):
        self.members = []

    def getBestCenter(self):
        if len(self.members) == 1:
            return self.center
        min_dis = 100000000
        best_c = -1
        for m in self.members:
            tot_dis = 0
            for n in self.members:
                if m == n:
                    continue
                tot_dis += (self.distance_matrix[m][n] + self.distance_matrix[n][m]) / 2
            if tot_dis < min_dis:
                min_dis = 0
                best_c = m
        assert best_c >= 0
        return best_c

    def getDistanceFromCenter(self, d):
        return (self.distance_matrix[self.center][d] + self.distance_matrix[d][self.center]) / 2

    def getWithinClusterDistance(self):
        tot_dis = 0
        for i in xrange(len(self.members)):
            dis = 0
            for j in xrange(len(self.members)):
                if i == j:
                    continue
                p_1 = self.members[i]
                p_2 = self.members[j]
                dis += (self.distance_matrix[p_1][p_2] + self.distance_matrix[p_2][p_1]) ** 2 / 4
            tot_dis += math.sqrt(dis / len(self.members))


        return tot_dis

def kmedoid_clustering(N, dm, k):
    # N : # of points, N = len(dm[0])
    # dm : distance matrix
    # k : # of cluster

    clusters = [Cluster(dm) for i in xrange(k)]

    centers = range(N)
    random.shuffle(centers)
    centers = set(centers[0:k])

    idx = 0
    for c in centers:
        clusters[idx].setCenter(c)
        idx += 1

    last_dis = 1000000000
    it = 0
    while True:
        # assignment
        for i in xrange(N):
            if i in centers:
                continue
            min_dis = 100000
            best_c = None
            for c in clusters:
                dis = c.getDistanceFromCenter(i)
                if dis < min_dis:
                    min_dis = dis
                    best_c = c
            best_c.addMember(i)

        it += 1
        if it > 100:
            break

        tot_dis = 0
        for c in clusters:
            tot_dis += c.getWithinClusterDistance()

        if math.fabs(tot_dis - last_dis) < 1e-8:
            break

        last_dis = tot_dis

        for c in clusters:
            best_c = c.getBestCenter()
            c.clearMember()
            c.setCenter(best_c)
    print "After %d iterations, converges" % it
    return clusters

def kmedoid_clustering_list(N, dm, k, L=None):
    # N : # of points, N = len(dm[0])
    # dm : distance matrix
    # k : # of cluster
    # L : list of points to cluster

    if L is None:
        L = range(N)

    assert k <= len(L)

    clusters = [Cluster(dm) for i in xrange(k)]

    centers = copy.deepcopy(L)
    random.shuffle(centers)
    centers = set(centers[0:k])

    idx = 0
    for c in centers:
        clusters[idx].setCenter(c)
        idx += 1

    last_dis = 1000000000
    it = 0
    while True:
        # assignment
        for i in L:
            if i in centers:
                continue
            min_dis = 100000
            best_c = None
            for c in clusters:
                dis = c.getDistanceFromCenter(i)
                if dis < min_dis:
                    min_dis = dis
                    best_c = c
            best_c.addMember(i)

        it += 1
        if it > 100:
            break

        tot_dis = 0
        for c in clusters:
            tot_dis += c.getWithinClusterDistance()

        if math.fabs(tot_dis - last_dis) < 1e-8:
            break

        last_dis = tot_dis

        for c in clusters:
            best_c = c.getBestCenter()
            c.clearMember()
            c.setCenter(best_c)
    # print "After %d iterations, converges" % it
    return clusters

def chooseBestClustering(N, dm, k, T, L=None):
    min_dis = 1000000
    err = []
    for t in xrange(T):
        clusters = kmedoid_clustering_list(N, dm, k, L)
        tot_dis = 0
        for c in clusters:
            tot_dis += c.getWithinClusterDistance()
        if tot_dis < min_dis:
            min_dis = tot_dis
            best_clusters = clusters
        err.append(min_dis)
    return best_clusters, err


def createData():
    N = 9
    dm = np.zeros((N, N))
    dm[0][1] = 0.5
    dm[1][0] = 0.5
    dm[0][2] = 0.9
    dm[2][0] = 0.5

    dm[3][4] = 0.5
    dm[4][3] = 0.5
    dm[3][5] = 0.5
    dm[5][3] = 0.5

    dm[6][7] = 0.5
    dm[7][6] = 0.5
    dm[6][8] = 0.5
    dm[8][6] = 0.5

    # dm is the affinity matrix
    return N, dm

def createData2():
    N = 300
    dm = np.zeros((N, N))
    for i in xrange(1000):
        n_1 = random.randint(0, (N-1)/3)
        n_2 = random.randint(0, (N-1)/3)
        while n_2 == n_1:
            n_2 = random.randint(0, (N-1)/3)
        w = random.random()
        dm[n_1][n_2] = w
        dm[n_2][n_1] = w

    for i in xrange(1000):
        n_1 = random.randint(N/3, 2*(N-1)/3)
        n_2 = random.randint(N/3, 2*(N-1)/3)
        while n_2 == n_1:
            n_2 = random.randint(N/3, 2*(N-1)/3)
        w = random.random()
        dm[n_1][n_2] = w
        dm[n_2][n_1] = w

    for i in xrange(40):
        n_1 = random.randint(2*N/3, N-1)
        n_2 = random.randint(2*N/3, N-1)
        while n_2 == n_1:
            n_2 = random.randint(2*N/3, N-1)
        w = random.random()
        dm[n_1][n_2] = w
        dm[n_2][n_1] = w

    return N, dm

def affinityToDistance(N, am):
    dm = np.zeros((N, N))
    for i in xrange(N):
        for j in xrange(N):
            dm[i][j] = 1 - am[i][j]
    return dm

def plitCluster(N, dm, cluster, M):
    if len(cluster.members) <= M:
        return [cluster]

    clusters, err = chooseBestClustering(N, dm, 2, 5, L=cluster.members)
    res = []
    for c in clusters:
        for sub_c in plitCluster(N, dm, c, M):
            res.append(sub_c)

    return res

if __name__ == '__main__':
    # N = 10
    # am = np.zeros((N, N))

    N, am = createData2()
    dm = affinityToDistance(N, am)

    clusters, err = chooseBestClustering(N, dm, 5, 20)
    for c in clusters:
        print c.members, c.getWithinClusterDistance()
    print err


