import numpy as np

from constant import *


def classifier(my_classifier, x_train_temp, x_test_temp, y_train_temp, y_test_temp):
    """
    Train a classifier on test data and return accuracy and prediction on test data

    :param my_classifier:
    :param x_train_temp:
    :param x_test_temp:
    :param y_train_temp:
    :param y_test_temp:

    :return: accuracy, prediction
    """
    # Fit the model on the training data.
    my_classifier.fit(x_train_temp, y_train_temp)

    # See how the model performs on the test data.
    accuracy = my_classifier.score(x_test_temp, y_test_temp)
    prediction = my_classifier.predict(x_test_temp)

    return accuracy, prediction


class EncoderDecoderNetwork:
    def __init__(
            self,
            input_channels,
            output_channels,
            hidden_layer_sizes=[1000, 500, 250],
            n_dims_code=125,
            learning_rate=0.001,
            activation_fn=tf.nn.elu,
    ):
        """
        Implement an encoder decoder network and train it

        :param input_channels: number of source robot features
        :param output_channels: number of target robot features
        :param hidden_layer_sizes: units in hidden layers
        :param n_dims_code: code vector length
        :param learning_rate: learning rate
        :param activation_fn: activation function
        """
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.hidden_layer_sizes = hidden_layer_sizes
        self.n_dims_code = n_dims_code
        self.learning_rate = learning_rate
        self.activation_fn = activation_fn

        self.X = tf.placeholder("float", [None, self.input_channels], name='InputData')
        self.Y = tf.placeholder("float", [None, self.output_channels], name='OutputData')

        self.code_prediction = self.encoder()
        self.output = self.decoder(self.code_prediction)

        # Define loss
        with tf.name_scope('Loss'):
            # Root-mean-square error (RMSE)
            self.cost = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(self.output, self.Y))))

        # Define optimizer
        with tf.name_scope('Optimizer'):
            self.train_op = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.cost)

        # 'Saver' op to save and restore all the variables
        self.saver = tf.train.Saver(max_to_keep=1)

        # Create a summary to monitor cost tensor
        tf.summary.scalar("loss", self.cost)

        # Merge all summaries into a single op
        self.merged_summary_op = tf.summary.merge_all()

        # Initializing the variables
        self.sess = tf.Session()  # tf.InteractiveSession()
        self.sess.run(tf.global_variables_initializer())

    def encoder(self):
        with tf.name_scope('Encoder'):
            for i in range(1, len(self.hidden_layer_sizes ) +1):
                if i == 1:
                    net = tf.layers.dense(inputs=self.X, units=self.hidden_layer_sizes[i-1], activation=self.activation_fn, name="encoder_" +str(i))
                else:
                    net = tf.layers.dense(inputs=net, units=self.hidden_layer_sizes[i-1], activation=self.activation_fn, name="encoder_" +str(i))
            # net = tf.layers.dense(inputs=net, units=self.n_dims_code, activation=self.activation_fn) # GT try this
            net = tf.layers.dense(inputs=net, units=self.n_dims_code)
        return net

    def decoder(self, net):
        with tf.name_scope('Decoder'):
            for i in range(len(self.hidden_layer_sizes), 0, -1):
                net = tf.layers.dense(inputs=net, units=self.hidden_layer_sizes[i-1], activation=self.activation_fn, name="decoder_" +str(i))
            net = tf.layers.dense(inputs=net, units=self.output_channels, name="decoder_final")
        return net

    def train_session(self, x_data, y_data, logs_path):
        """
        Train using provided data

        :param x_data: source robot features
        :param y_data: target robot features
        :param logs_path: log path

        :return: cost over training
        """

        x_data = x_data.reshape(-1, self.input_channels)
        y_data = y_data.reshape(-1, self.output_channels)

        # Write logs to Tensorboard
        if logs_path is not None:
            summary_writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())

        cost_log = []
        # Start Training
        for epoch in range(TRAINING_EPOCHS):
            # Run optimization op (backprop), cost op (to get loss value)
            _, c = self.sess.run([self.train_op, self.cost], feed_dict={self.X: x_data, self.Y: y_data})

            cost_log.append(c)

            # Print generated data after every 100 epoch
            # if (epoch + 1) % 100 == 0:
            #     print("Epoch:", '%04d' % (epoch+1), "cost =", "{:.9f}".format(c))
            #     generated_output = self.sess.run(self.output, feed_dict={self.X: x_data})
            #     print("Generated: ")
            #     print(list(generated_output[0]))
            #     print("Original: ")
            #     print(list(y_data[0]))

            # Write logs at every iteration
            if logs_path is not None:
                summary = self.sess.run(self.merged_summary_op, feed_dict={self.X: x_data, self.Y: y_data})
                summary_writer.add_summary(summary, epoch)

        return cost_log

    def generate(self, x_data):
        """
        Generate target robot data using source robot data

        :param x_data: source robot data

        :return: generated target robot data
        """

        x_data = x_data.reshape(-1, self.input_channels)
        generated_output = self.sess.run(self.output, feed_dict={self.X: x_data})

        return generated_output

    def rmse_loss(self, x_data, y_data):
        """
        Return the Root mean square error

        :param x_data:
        :param y_data:

        :return:
        """
        loss = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(x_data, y_data))))
        loss = self.sess.run(loss)

        #np_loss = np.sqrt(np.mean(np.square(np.subtract(x_data, y_data))))

        return loss


class EncoderDecoderNetwork_b_VAE:
    def __init__(
            self,
            num_of_domains,
            num_of_features,
            domain_names,
            activation_fn,
            beta = 1,
            hidden_layer_sizes=[1000, 500, 250],
            learning_rate=0.0001,
            training_epochs=1000,
    ):
        """
        Implement an beta auto encoder network and train it

        :param num_of_domains: number of domains
        :param num_of_features: a list of number of features in each domain
        :param domain_names: domain names
        :param activation_fn: activation function
        :param beta: beta
        :param hidden_layer_sizes: units in hidden layers
        :param n_dims_code: code vector length
        :param learning_rate: learning rate
        :param training_epochs: training epochs
        """

        self.num_of_domains = num_of_domains
        self.num_of_features = num_of_features
        self.domain_names = domain_names
        self.activation_fn = activation_fn
        self.domain_names = domain_names
        self.beta = beta
        self.hidden_layer_sizes = hidden_layer_sizes
        self.learning_rate = learning_rate
        self.training_epochs = training_epochs

        self.placeholder = {"input":[], "output":[], "prediction": []}

        for a_domain in range(self.num_of_domains):
            self.placeholder["input"].append(tf.placeholder("float", [None, self.num_of_features[a_domain]], name='input_'+str(a_domain)))
            self.placeholder["output"].append(tf.placeholder("float", [None, self.num_of_features[a_domain]], name='output_'+str(a_domain)))
        
        # Encoders
        encoder_outputs = self.encoder(self.placeholder["input"][0], self.domain_names[0])
        for a_domain in range(1, self.num_of_domains):
            a_encoder_output = self.encoder(self.placeholder["input"][a_domain], self.domain_names[a_domain])
            encoder_outputs= tf.concat([encoder_outputs, a_encoder_output], axis=1, name='encoder_outputs')
        
        self.code_prediction, self.z_mu, self.z_log_sigma_sq = self.latent_code(encoder_outputs)
        
        # Decoders
        for a_domain in range(self.num_of_domains):
            a_decoder_output = self.decoder(self.code_prediction, self.num_of_features[a_domain], self.domain_names[a_domain])
            self.placeholder["prediction"].append(a_decoder_output)
        
        # Reconstruction cost
        # concatenating all the domains to optimize them together
        prediction_concat = self.placeholder["prediction"][0]
        output_concat = self.placeholder["output"][0]
        for a_domain in range(1, self.num_of_domains):
            prediction_concat = tf.concat([prediction_concat, self.placeholder["prediction"][a_domain]], axis=1, name='prediction_concat')
            output_concat = tf.concat([output_concat, self.placeholder["output"][a_domain]], axis=1, name='output_concat')
        
        #recon_loss = -tf.reduce_sum(output_concat * tf.log(1e-10+prediction_concat) + (1-output_concat) * tf.log(1e-10+1-prediction_concat), axis=1)
        #recon_loss = tf.reduce_mean(recon_loss)
        
        recon_loss = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(prediction_concat, output_concat))))
        
        # Latent loss
        # Kullback Leibler divergence: measure the difference between two distributions
        # Here we measure the divergence between the latent distribution and N(0, 1)
        kl_penalty = -0.5 * tf.reduce_sum(1 + self.z_log_sigma_sq - tf.square(self.z_mu) - tf.exp(self.z_log_sigma_sq), axis=1)
        kl_penalty = tf.reduce_mean(kl_penalty)

        self.cost = tf.reduce_mean(recon_loss + self.beta * kl_penalty)
        
        self.train_op = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.cost)

        # 'Saver' op to save and restore all the variables
        self.saver = tf.train.Saver(max_to_keep=1)

        # Create a summary to monitor cost tensor
        tf.summary.scalar("loss", self.cost)

        # Merge all summaries into a single op
        self.merged_summary_op = tf.summary.merge_all()

        # Initializing the variables
        self.sess = tf.Session()  # tf.InteractiveSession()
        self.sess.run(tf.global_variables_initializer())


    def encoder(self, X, domain_name):
        for i in range(1, len(self.hidden_layer_sizes)+1):
            if i == 1:
                net = tf.layers.dense(inputs=X, units=self.hidden_layer_sizes[i-1], activation=self.activation_fn, name="Encoder_"+domain_name+"_layer_"+str(i))
                #print(net)
            else:
                net = tf.layers.dense(inputs=net, units=self.hidden_layer_sizes[i-1], activation=self.activation_fn, name="Encoder_"+domain_name+"_layer_"+str(i))
                #print(net)
        return net

    def latent_code(self, X):
        _, encoder_len = X.get_shape()
        z_mu = tf.layers.dense(inputs=X, units=encoder_len, activation=None, name='z_mu')
        z_log_sigma_sq = tf.layers.dense(inputs=X, units=encoder_len, activation=None, name='z_log_sigma_sq')
        eps = tf.random_normal(shape=tf.shape(z_log_sigma_sq), mean=0, stddev=1, dtype=tf.float32)
        z = z_mu + tf.sqrt(tf.exp(z_log_sigma_sq)) * eps # The reparameterization trick
        
        return z, z_mu, z_log_sigma_sq

    def decoder(self, net, domain_size, domain_name):
        for i in range(len(self.hidden_layer_sizes), 0, -1):
            net = tf.layers.dense(inputs=net, units=self.hidden_layer_sizes[i-1], activation=self.activation_fn, name="Decoder_"+domain_name+"_layer_"+str(i))
            #print(net)
        #net = tf.layers.dense(inputs=net, units=domain_size, activation=tf.sigmoid, name="Decoder_Final_"+domain_name) # For MNIST, pixels are between 0 & 1
        net = tf.layers.dense(inputs=net, units=domain_size, activation=None, name="Decoder_Final_"+domain_name)
        #print(net)
        return net
    
    def get_feed_dict(self, domains_data_input, domains_data_output):
        feed_dict = {}
        for a_domain in range(self.num_of_domains):
            #print(a_domain)
            feed_dict[self.placeholder["input"][a_domain]] = domains_data_input['domain_'+str(a_domain)]
            feed_dict[self.placeholder["output"][a_domain]] = domains_data_output['domain_'+str(a_domain)]
        return feed_dict

    def train_session(self, domains_data_train, logs_path):
        """
        Train using provided data

        :param feed_dict_train: feed_dict_train
        :param logs_path: log path

        :return: cost over training
        """

        # Write logs to Tensorboard
        if logs_path is not None:
            summary_writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())
        
        #data_drop = np.random.randint(low=0, high=self.num_of_domains, size=self.training_epochs)
        data_drop = np.random.randint(low=1, high=2, size=self.training_epochs) # No Data
        
        cost_log = []
        # Start Training
        for epoch in range(self.training_epochs):
            data_train = domains_data_train.copy()
            if data_drop[epoch] == 0:
                data_train['domain_'+str(self.num_of_domains-1)] = data_train['domain_'+str(self.num_of_domains-1)]
                #print("Full Data")
            elif data_drop[epoch] == 1:
                data_train['domain_'+str(self.num_of_domains-1)] = np.zeros(data_train['domain_'+str(self.num_of_domains-1)].shape)
                #print("No Data")
            elif data_drop[epoch] == 2:
                num_of_examples = data_train['domain_'+str(self.num_of_domains-1)].shape[0]
                examples_to_drop = np.random.randint(low=0, high=num_of_examples, size=num_of_examples//2)
                data_train['domain_'+str(self.num_of_domains-1)] = [data_train['domain_'+str(self.num_of_domains-1)][i] * 0 if i in examples_to_drop else data_train['domain_'+str(self.num_of_domains-1)][i] for i in range(num_of_examples)]
                data_train['domain_'+str(self.num_of_domains-1)] = np.array(data_train['domain_'+str(self.num_of_domains-1)])
                #print("Some Data")
                
            feed_dict_train = self.get_feed_dict(data_train, domains_data_train)
            # Run optimization op (backprop), cost op (to get loss value)
            _, c = self.sess.run([self.train_op, self.cost], feed_dict=feed_dict_train)

            cost_log.append(c)

            # Write logs at every iteration
            if logs_path is not None:
                summary = self.sess.run(self.merged_summary_op, feed_dict={self.X: x_data, self.Y: y_data})
                summary_writer.add_summary(summary, epoch)

        return cost_log

    def rmse_loss(self, x_data, y_data):
        """
        Return the Root mean square error

        :param x_data:
        :param y_data:

        :return:
        """
        loss = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(x_data, y_data))))
        loss = self.sess.run(loss)

        #np_loss = np.sqrt(np.mean(np.square(np.subtract(x_data, y_data))))

        return loss


class EncoderDecoderNetwork_b_VEDN:
    def __init__(
            self,
            input_channels,
            output_channels,
            beta = 1,
            hidden_layer_sizes=[1000, 500, 250],
            n_dims_code=125,
            learning_rate=0.001,
            activation_fn=tf.nn.elu,
            training_epochs=1000,
    ):
        """
        Implement an Beta variational encoder decoder network and train it

        :param input_channels: number of source robot features
        :param output_channels: number of target robot features
        :param hidden_layer_sizes: units in hidden layers
        :param n_dims_code: code vector length
        :param learning_rate: learning rate
        :param activation_fn: activation function
        """
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.beta = beta
        self.hidden_layer_sizes = hidden_layer_sizes
        self.n_dims_code = n_dims_code
        self.learning_rate = learning_rate
        self.activation_fn = activation_fn
        self.training_epochs = training_epochs

        self.X = tf.placeholder("float", [None, self.input_channels], name='InputData')
        self.Y = tf.placeholder("float", [None, self.output_channels], name='OutputData')

        self.code_prediction, self.z_mu, self.z_log_sigma_sq = self.encoder()
        self.output = self.decoder(self.code_prediction)

        # Define loss
        with tf.name_scope('Loss'):
            # Root-mean-square error (RMSE)
            self.cost = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(self.output, self.Y))))
            
            # Latent loss
            # Kullback Leibler divergence: measure the difference between two distributions
            # Here we measure the divergence between the latent distribution and N(0, 1)
            self.latent_loss = -0.5 * tf.reduce_sum(1 + self.z_log_sigma_sq - tf.square(self.z_mu) - tf.exp(self.z_log_sigma_sq), axis=1)
            #print("latent_loss: ", self.latent_loss)
            self.latent_loss = tf.reduce_mean(self.latent_loss)
            #print("latent_loss: ", self.latent_loss)

            self.cost = tf.reduce_mean(self.cost + self.beta *self.latent_loss)
            #print("cost: ", self.cost)

        # Define optimizer
        with tf.name_scope('Optimizer'):
            self.train_op = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.cost)

        # 'Saver' op to save and restore all the variables
        self.saver = tf.train.Saver(max_to_keep=1)

        # Create a summary to monitor cost tensor
        tf.summary.scalar("loss", self.cost)

        # Merge all summaries into a single op
        self.merged_summary_op = tf.summary.merge_all()

        # Initializing the variables
        self.sess = tf.Session()  # tf.InteractiveSession()
        self.sess.run(tf.global_variables_initializer())

    def encoder(self):
        with tf.name_scope('Encoder'):
            for i in range(1, len(self.hidden_layer_sizes)+1):
                if i == 1:
                    net = tf.layers.dense(inputs=self.X, units=self.hidden_layer_sizes[i-1], activation=self.activation_fn, name="encoder_"+str(i))
                else:
                    net = tf.layers.dense(inputs=net, units=self.hidden_layer_sizes[i-1], activation=self.activation_fn, name="encoder_"+str(i))
            #net = tf.layers.dense(inputs=net, units=self.n_dims_code, activation=self.activation_fn)
            z_mu = tf.layers.dense(inputs=net, units=self.n_dims_code, activation=None, name='z_mu')
            z_log_sigma_sq = tf.layers.dense(inputs=net, units=self.n_dims_code, activation=None, name='z_log_sigma_sq')
            eps = tf.random_normal(shape=tf.shape(z_log_sigma_sq), mean=0, stddev=1, dtype=tf.float32)
            z = z_mu + tf.sqrt(tf.exp(z_log_sigma_sq)) * eps # The reparameterization trick
        return z, z_mu, z_log_sigma_sq

    def decoder(self, net):
        with tf.name_scope('Decoder'):
            for i in range(len(self.hidden_layer_sizes), 0, -1):
                net = tf.layers.dense(inputs=net, units=self.hidden_layer_sizes[i-1], activation=self.activation_fn, name="decoder_"+str(i))
            net = tf.layers.dense(inputs=net, units=self.output_channels, activation=None, name="decoder_final")
        return net

    def train_session(self, x_data, y_data, logs_path):
        """
        Train using provided data

        :param x_data: source robot features
        :param y_data: target robot features
        :param logs_path: log path

        :return: cost over training
        """

        x_data = x_data.reshape(-1, self.input_channels)
        y_data = y_data.reshape(-1, self.output_channels)

        # Write logs to Tensorboard
        if logs_path is not None:
            summary_writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())

        cost_log = []
        # Start Training
        for epoch in range(self.training_epochs):
            # Run optimization op (backprop), cost op (to get loss value)
            _, c = self.sess.run([self.train_op, self.cost], feed_dict={self.X: x_data, self.Y: y_data})

            cost_log.append(c)

            # Print generated data after every 100 epoch
            # if (epoch + 1) % 100 == 0:
            #     print("Epoch:", '%04d' % (epoch+1), "cost =", "{:.9f}".format(c))
            #     generated_output = self.sess.run(self.output, feed_dict={self.X: x_data})
            #     print("Generated: ")
            #     print(list(generated_output[0]))
            #     print("Original: ")
            #     print(list(y_data[0]))

            # Write logs at every iteration
            if logs_path is not None:
                summary = self.sess.run(self.merged_summary_op, feed_dict={self.X: x_data, self.Y: y_data})
                summary_writer.add_summary(summary, epoch)

        return cost_log

    def generate_code(self, x_data):
        """
        Generate target robot data using source robot data

        :param x_data: source robot data

        :return: generated target robot data
        """

        x_data = x_data.reshape(-1, self.input_channels)
        generated_code = self.sess.run(self.code_prediction, feed_dict={self.X: x_data})

        return generated_code

    def generate(self, x_data):
        """
        Reconstruct input by passing through encoder and decoder

        :param x_data: input data

        :return: generated input data
        """

        x_data = x_data.reshape(-1, self.input_channels)
        generated_output = self.sess.run(self.output, feed_dict={self.X: x_data})

        return generated_output

    def rmse_loss(self, x_data, y_data):
        """
        Return the Root mean square error

        :param x_data:
        :param y_data:

        :return:
        """
        loss = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(x_data, y_data))))
        loss = self.sess.run(loss)

        #np_loss = np.sqrt(np.mean(np.square(np.subtract(x_data, y_data))))

        return loss

