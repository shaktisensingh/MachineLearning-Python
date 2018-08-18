import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

def main():
    file_name = 'ex7data2.mat'
    nos = 3  # number of clusters
    itr = 10  # number of iterations
    color = np.array(['m', 'k', 'g', 'c', 'y', 'r', 'b'])
    data = read_data(file_name)
##    plot_data(data)
    c = get_random_cluster_points(data, nos)
    print('C initialization',c)
    for i in range(itr):
        idx = assign_cluster(data, c)
        c = recalculate_centriods(data, idx, nos)
    print('c after claculation',c)
    plot_clusters(data, idx, nos, color)
    plot_data(c, s = 20)
    return

def plot_clusters(data, idx, nos, color):
    idx = idx[:,None]
    data = np.hstack((data,idx))
    for i in range(nos):
        temp = data[data[:,2] == i]
        plot_data(temp)
    return
    
def recalculate_centriods(data, idx, nos):
    c = np.zeros((nos, 2))
    for i in range(nos):
        temp = (idx == i)[:,None]
        c[i,:] = np.sum((data*temp), axis = 0, keepdims = True) / np.sum(temp)
    return c

def assign_cluster(data, c):
    distance = np.zeros(c.shape[0])
    idx = np.array([])
    for i in range(data.shape[0]):
        for j in range(c.shape[0]):
            distance[j] = np.sum(np.square(data[i,:] - c[j,:]))
        idx = np.append(idx, np.argmin(distance))
    return idx

def get_random_cluster_points(data, nos):
    c_x = np.random.randint(np.min(data[:,0],axis=0), np.max(data[:,0],axis=0),nos)[:,None]
    c_y = np.random.randint(np.min(data[:,1],axis=0), np.max(data[:,1],axis=0),nos)[:,None]
    return np.hstack((c_x,c_y))
    
def plot_data(data, s=5):
    plt.scatter(data[:,0], data[:,1],s=s)
    plt.show(block = False)
    return

def read_data(file_name):
    data = sio.loadmat(file_name)
    return data['X']

if __name__=='__main__':
    main()
