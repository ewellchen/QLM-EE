

def cal_entanglment(model, layer_aim = 21, n_gram = 2, word_dim = 10, data = dev_data[1], len_sentence = 100):
	e_dim = np.pow(word_dim, n_gram)
	len_dev = len(data)
	get_layer_output = K.function([model.layers[0].input], model.layers[layer_aim].output)
	word_q_embeddings = []
	for k in range(len_dev):
		layer_output = get_layer_output([np.reshape(data[k],[1,len_sentence])])
		word_q_e = np.array(layer_output)
		word_q_embeddings.append(word_q_e)
	word_q_embeddings = np.array(word_q_embeddings)
	word_q_embeddings = np.reshape(word_q_embeddings,[len_dev,2, len_sentence, e_dim])
	word_q_embeddings_r =  np.reshape(word_q_embeddings[:,0,:,:],(len_dev*len_sentence*e_dim))
	word_q_embeddings_i =  np.reshape(word_q_embeddings[:,1,:,:],(len_dev*len_sentence*e_dim))
	word_q_embeddings_complex = []
	for i in range(len(flar)):
		word_q_embeddings_complex.append(complex(word_q_embeddings_r[i],word_q_embeddings_i[i]))
	word_q_embeddings_complex = np.reshape(word_q_embeddings_complex,(len_dev,len_sentence,e_dim))

	entanglement_degree = []
	for i in range(len_dev):
		entangle_degree = []
		for j in range(len_sentence):
			b = np.reshape(word_q_embeddings_complex[i,j],[word_dim,word_dim])
			alpha = pyqentangle.schmidt_decomposition(b)
			entangle_d = 0
			for k in range(10):
				entangle_d = entangle_d - pow(alpha[k][0], 2) * np.log(pow(alpha[k][0], 2))
			entangle_degree.append(entangle_d)
		entanglement_degree.append(entangle_degree)
	entanglement_degree = np.array(entanglement_degree)
	entanglement_degree_all = np.reshape(entanglement_degree,[len_dev*len_sentence])
	heapq.nlargest(100,range(len(entanglement_degree_all)), entanglement_degree_all.take)
