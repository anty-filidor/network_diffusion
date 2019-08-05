import networkx as nx
from network_diffusion import flat_spreading as fs

M = nx.les_miserables_graph()
# M = nx.barabasi_albert_graph(200, 50)

# Teaser of the SI diffusion
'''
list_S, list_I, list_iter, nodes_infected, par = fs.si_diffusion(M, fract_I=0.05, beta_coeff=0.2,
                                                                 name='Les_miserables_V_Hugo_graph')

fig, ax = plt.subplots(1)
plt.plot(list_iter, list_S, label='suspected')
plt.plot(list_iter, list_I, label='infected')
plt.title('SI diffusion')
plt.legend()
plt.grid()
plt.savefig("{}.png".format(par[0]), dpi=150)
plt.show()

fs.visualise_si_nodes(M, nodes_infected, par)
fs.visualise_si_nodes_edges(M, nodes_infected, par)
'''

# Teaser of the SIR diffusion

list_S, list_I, list_R, list_iter, nodes_infected, nodes_recovered, par = \
    fs.sir_diffusion(M, fract_I=0.08, beta_coeff=0.2, gamma_coeff=0.2, name='Les_miserables_W_Hugo_graph')
'''
plt.plot(list_iter, list_S, label='suspected')
plt.plot(list_iter, list_I, label='infected')
plt.plot(list_iter, list_R, label='recovered')
plt.title('SIR diffusion')
plt.legend()
plt.grid()
plt.savefig("{}.png".format(par[0]), dpi=150)
plt.show()

fs.visualise_sir_nodes(M, nodes_infected, nodes_recovered, par)
fs.visualise_sir_nodes_edges(M, nodes_infected, nodes_recovered, par)
'''
