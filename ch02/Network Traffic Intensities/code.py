import numpy as np
from mininet.cli import CLI
from mininet.log import setLogLevel
from mininet.net import Mininet
from mininet.topo import Topo
from mininet.link import TCLink
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from decimal import *
import sys
import time

##check round-decimal

class customTopo(Topo):
    def build( self , rtt_delays , **params ):
        S = Decimal('40')
        host = self.addHost('h0')
        switch = self.addSwitch('s0')
        self.addLink(host, switch, cls=TCLink, delay='0ms', loss=0 , jitter = '0ms')
        for n in range(int(S)):
            self.addNode('n%s'%(n+Decimal('1')))
        print('node:  ' + str(self.nodes()))
        delays = [20,14,10,13,10,18,4,8,18,16,4,7,2,4,8,11,11,8,3,4,2,13,17,20,1,7,2,4,11,18,12,20,6,11,15,10,19,11,4,20]
        jitters = [0.7839,3.0438,2.3883,3.6728,4.8108,0.7595,1.8332,3.1177,1.801,2.3787,0.9,0.0013,1.1,1.4042,3.005,3.3607,4.6659,2.0181,0.2978,0.7612,0.4714,2.6111,2.398,3.1771,0.005,4,0.7,1.0520,0.12,3.6648 , 0.696 , 3.6662 , 0.432 , 2.1297,4.3,1.9505,3.9585,3.8328,0.0688,5.9310]
        for n in range(int(S)):
            rtt_delays.append(2*delays[n])
            self.addLink(switch, self.nodes()[n + 1], cls=TCLink, delay=str(delays[n])+'ms', loss=0 , jitter = str(jitters[n])+'ms')



def create_network(cov_samples_per_rv):
    S = Decimal('40')
    rtt_delays = []
    topo = customTopo(rtt_delays = rtt_delays)
    net = Mininet(topo = topo , waitConnected=True)
    net.start()
    print('\n \n \n *** \t network is created \t *** \n \n \n')
    samples = []
    sigma_temp = []
    host = net.getNodeByName('h0')
    for i in range(int(S)):
        node = net.getNodeByName('n%s'%(i + 1))
        temp = []
        for p in range(cov_samples_per_rv):
            res = host.cmd('ping -c1', node.IP())
            failed = False
            fail_counter = Decimal('0')
            try:
                temp.append(float(res.split("/")[-3]))
            except:
                failed = True
                print('\npacket loss on node n%s'%(i + 1)+'. trying again...\n')
                fail_counter += Decimal('1')
            while(failed):
                if fail_counter == Decimal('100'):
                    sys.exit('doomed link :) exiting the program')
                res = host.cmd('ping -c1', node.IP())
                try:
                    temp.append(float(res.split("/")[-3]))
                    failed = False
                except:
                    print('\npacket loss on node n%s'%(i + 1)+'. trying again...\n')
                    fail_counter += Decimal('1')
            sigma_temp.append(temp[-1])
        print('node%s' % (i + 1) + ' samples are captured...')
        samples.append(temp.copy())
    return net, np.cov(samples, rowvar=True, bias=True), Decimal(str(np.std(sigma_temp))) , rtt_delays




def get_D(Pi, N, K, sigma :Decimal):
    K_inv = np.linalg.inv(K)
    W = np.diag(Pi * N * np.power(sigma , Decimal('-2')))
    t = np.linalg.inv(K_inv.astype(float) + W.astype(float))
    return t



def arg_min_nonzero(d, w):
    if np.max(w) <= Decimal('0'):
        sys.exit('wrong weights!')
    tempmat = np.diagonal(d).copy()
    index = np.argmin(tempmat)
    while w[index] == Decimal('0'):
        tempmat[index] = np.max(tempmat)
        #no longer the minimum!
        index = np.argmin(tempmat)
    return index



def alg(N:Decimal , K , sigma :Decimal , gamma:Decimal , epsilon:Decimal , uniform_initialization = True):
    S = Decimal('40')
    weights =  []
    alpha_t = -1
    if uniform_initialization or N < S:
        for i in range(int(S)):
            weights.append(Decimal('1') / S)
        weights = np.array(weights)
        alpha_t = Decimal('1') / S
    else:
        alpha_t = Decimal('1') / N
        for i in range(int(S)):
            weights.append(Decimal('1')/N)
        weights = np.array(weights)
        covmat_diagonal_values = np.diagonal(K)
        probes = covmat_diagonal_values/np.sum(covmat_diagonal_values)
        for i in range(int(N - S)):
            index = np.random.choice(int(S) , size=1 , p = probes.astype(float))
            weights[index] += Decimal('1')/N

    Dt = get_D(weights, N, K, sigma)
    itr = Decimal('-1')
    while True:
        itr += Decimal('1')
        number_of_nonzeros = Decimal('0')
        for i in range(len(weights)):
            if not weights[i] == Decimal('0'):
                number_of_nonzeros += Decimal('1')
        if alpha_t < epsilon or number_of_nonzeros <= Decimal('2'):
            return weights
        i = np.argmax(np.diagonal(Dt))
        weights[i] += alpha_t
        ###check later
        # if i == j:
        #     i = np.random.randint(0 , len(weights))
        #     while(weights[j] == 0):
        #         j = np.random.randint(0 , len(weights))
        ###......
        Dtp = get_D(weights, N, K, sigma)
        j = arg_min_nonzero(Dtp, weights)
        if weights[j] - alpha_t < Decimal('0'):
            print('negative weight')
            print('current weight:  ' + str(weights[j]))
            print('alpha_t:  ' + str(alpha_t))
            weights[j] = Decimal('0')
        else:
            weights[j] -= alpha_t
        Dtt = get_D(weights, N, K, sigma)
        if not Decimal(str(np.linalg.det(Dt))) == Decimal('0'):
            if not Decimal(str(np.linalg.det(Dtt))) / Decimal(str(np.linalg.det(Dt))) < Decimal('1') - (gamma):
                alpha_t /= Decimal('2')
        else:
            print('Error: Determinant of matrix D is 0')
            #return weights
        Dt = Dtt


def estimate(net , N:Decimal , covariance_matrix , sigma:Decimal , estimation_Type = 'uniform' , nodes_to_monitor = None , weights = None):
    avg_results = []
    if estimation_Type == 'uniform':
        weights = []
        for i in range(len(covariance_matrix[0])):
            weights.append(Decimal('1')/Decimal(str(len(covariance_matrix[0]))))
        weights = np.array(weights)
        nodes_to_monitor = range(len(covariance_matrix[0]))
        if N < Decimal(str(len(covariance_matrix[0]))):
            nodes_to_monitor = np.random.choice(len(covariance_matrix[0]) , size=int(N) , replace=False)
            weights = []
            for i in range(len(covariance_matrix[0])):
                weights.append(Decimal('0'))
            weights = np.array(weights)
            weights[nodes_to_monitor] = Decimal('1')/N
        #print('sum of weights:  ' + str(np.sum(weights)))
        host = net.getNodeByName('h0')
        for i in range(len(nodes_to_monitor)):
            node = net.getNodeByName('n%s' % (nodes_to_monitor[i] + 1))
            temp = []
            for j in range(round(N / Decimal(str(len(nodes_to_monitor))))):
                res = host.cmd('ping -c1', node.IP())
                failed = False
                fail_counter = Decimal('0')
                try:
                    temp.append(Decimal(res.split("/")[-3]))
                except:
                    failed = True
                    print('\npacket loss on node n%s'%(nodes_to_monitor[i] + 1)+'. trying again...\n')
                    fail_counter += Decimal('1')
                while(failed):
                    if fail_counter == Decimal('100'):
                        sys.exit('doomed link :) exiting the program')
                    res = host.cmd('ping -c1', node.IP())
                    try:
                        temp.append(Decimal(res.split("/")[-3]))
                        failed = False
                    except:
                        print('\npacket loss on node n%s'%(nodes_to_monitor[i] + 1)+'. trying again...\n')
                        fail_counter += Decimal('1')
            avg_results.append(np.mean(temp))
    else:
        sumval = Decimal('0')
        host = net.getNodeByName('h0')
        for i in range(len(nodes_to_monitor)):
            node = net.getNodeByName('n%s' % (nodes_to_monitor[i] + 1))
            sumval += round(N * weights[nodes_to_monitor[i]])
            temp = []
            for j in range(round(N * weights[nodes_to_monitor[i]])):
                res = host.cmd('ping -c1', node.IP())
                failed = False
                fail_counter = Decimal('0')
                try:
                    temp.append(Decimal(res.split("/")[-3]))
                except:
                    failed = True
                    print('\npacket loss on node n%s'%(nodes_to_monitor[i] + 1)+'. trying again...\n')
                    fail_counter += Decimal('1')
                while(failed):
                    if fail_counter == Decimal('100'):
                        sys.exit('doomed link :) exiting the program')
                    res = host.cmd('ping -c1', node.IP())
                    try:
                        temp.append(Decimal(res.split("/")[-3]))
                        failed = False
                    except:
                        print('\npacket loss on node n%s'%(nodes_to_monitor[i] + 1)+'. trying again...\n')
                        fail_counter += Decimal('1')
            avg_results.append(np.mean(temp))
        #print('supposed N :  ' + str(sumval))
    K_Z_ep = np.array(covariance_matrix[np.array(nodes_to_monitor), :])
    K_ep = np.array(covariance_matrix[np.array(nodes_to_monitor), :][:, np.array(nodes_to_monitor)])
    W_ep = np.array(np.diag(np.array(weights)[np.array(nodes_to_monitor)] * N * np.power(sigma , -2)))
    avg_results = np.array(avg_results).reshape(len(avg_results) , 1)
    U_estimation = np.matmul(np.matmul(K_Z_ep.T.astype(float),np.linalg.inv(K_ep.astype(float) + np.linalg.inv(W_ep.astype(float)))),avg_results.astype(float))
    # K_Z_ep = covariance_matrix[np.array(nodes_to_monitor), :][:, np.arange(S)].reshape(len(nodes_to_monitor), S)
    # K_ep = covariance_matrix[np.array(nodes_to_monitor), :][:, np.array(nodes_to_monitor)]
    # W_ep = np.diag(weights * N * sigma ** (-2.0))[np.array(nodes_to_monitor), :][:, np.array(nodes_to_monitor)]
    # U_estimation = np.matmul(np.matmul(K_Z_ep.T,np.linalg.inv((K_ep + np.linalg.inv(W_ep)))),(np.array(avg_results).reshape(len(avg_results) , 1)))
    # if estimation_Type == 'uniform':
    #     print('u est:  ' + str(U_estimation))
    # else:
    #     print('w est:  ' + str(U_estimation))
    return U_estimation

# def progressBar(current , total , barlength = 20):
#     percent = float(current)*100 / total
#     arrow = '-' * int(percent/100 * barlength -1) + '>'
#     spaces = ' ' * (barlength - len(arrow))
#
#     print('Progress: [%s%s] %d %%' % (arrow,spaces,percent) , end='r')
if __name__ == '__main__':

    cov_per = 50
    N = Decimal('80')
    gamma = Decimal('0.0000001')
    epsilon = Decimal('0.00000000001')
    uniform_init = False


    setLogLevel('info')
    net, covmat, sigma , rtt_delays  = create_network(cov_samples_per_rv=cov_per)
    # print('rtt delays =  ' + str(rtt_delays))
    # print('first entry:  ' + str(rtt_delays[0]))
    # print('first entry type:  ' + str(type(rtt_delays[0])))
    print('\n\n')
    print('10 runs of estimation using uniform weights...please wait...')
    uniform_estimations = []
    for i in range(10):
        msg = '\r \tprogress: %s'%((i)*10) +'%'
        sys.stdout.write(msg)
        uniform_estimations.append(estimate(net=net, N=N, covariance_matrix=covmat, sigma=sigma, estimation_Type='uniform'))
        sys.stdout.flush()
    print('\r \t uniform weights runs are done...')
    print('10 runs of estimation using algorithm...please wait...')
    weights_estimations = []
    supporting_points_list = []
    for i in range(10):
        # observations = []
        msg = '\r \tprogress: %s'%((i)*10) +'%'
        sys.stdout.write(msg)
        weights = alg(N=N, K=covmat, sigma=sigma, gamma=gamma, epsilon=epsilon , uniform_initialization=uniform_init)
        # for nn in range(len(weights)):
        #     observations.append(round(N*weights[nn]))
        # print('observations:  ' + str(observations))
        #counter = Decimal('0')
        nodes_to_monitor = []
        for i in range(len(weights)):
            # if weights[i] < Decimal('0'):
            #     print('negative weight!!!!')
            if round(N * weights[i]) > 0:
                nodes_to_monitor.append(i)
            # else:
            #     counter += Decimal('1')
        supporting_points_list.append('n='+str(len(nodes_to_monitor)))
        # print('number of zeros : ' + str(counter))
        weights_estimations.append(estimate(net=net, N=N, covariance_matrix=covmat,sigma=sigma, estimation_Type='using_weights' , nodes_to_monitor=nodes_to_monitor, weights=weights))
        sys.stdout.flush()
    print('\r \t algorithm runs are done...')
    print('\n\n')
    uniform_mses = []
    weights_mses = []
    for i in range(10):
        uniform_mses.append(mean_squared_error(uniform_estimations[i] , rtt_delays))
        weights_mses.append(mean_squared_error(weights_estimations[i] , rtt_delays))
    # print('sigma:   ' +str(sigma))
    print('uniform mses:   ' + str(uniform_mses))
    print('weighted mses:   ' + str(weights_mses))
    plt.plot(uniform_mses, 'o-', label='uniform')
    plt.plot(weights_mses, 'o-', label='using algorithm')
    plt.ylabel('MSE')
    plt.xlabel('runs(N='+str(N)+' , S='+str(40)+ ' , ' + r'$\sigma$'+'='+'%0.4f'%sigma+' , '+r'$\gamma$'+'='+str(gamma)+')')
    for i , txt in enumerate(supporting_points_list):
        plt.annotate(txt , (i , weights_mses[i]) , rotation=45)
    plt.legend()
    plt.savefig('results.jpg')
    plt.show()
    #CLI(net)
    net.stop()