import pandas as pd
import os
import numpy as np
import time
from torch.utils.tensorboard import SummaryWriter

class SaveResults:
    def __init__(self, model_args, training_args): 
        super().__init__()
        self.model_args = model_args
        self.training_args = training_args
        self.output_dir=self.training_args.output_dir + "/tracking"
        self.tensorboard_path=self.training_args.output_dir + "/" + self.model_args.tensorboard_path 
        self.save_name_pre = "empty"
        self.results = {'train_loss': [],  
                   'train_cov_loss': [], 
                   'train_sim_loss': [],
                   'train_weighted_cov_loss':[],
                   'train_weighted_sim_loss':[],
                   'train_cross_entropy':[],
                   'train_cos_sim':[],
                   'cls_output_max':[],
                   'cls_output_mean':[],
                   'cls_output_std':[]}

        self.called_times = 1
        self.save_frequency = model_args.tensorboard_save_frequency
        self.last_n_cov_losses = []
        self.last_n_sim_losses = []
        self.last_n_losses = []
        self.last_n_cross_entropy_losses = []
        self.last_n_cos_sim_losses = []
        self.last_n_cls_outputs = []

    def create_results(self):
        model_args = self.model_args
        training_args = self.training_args
        unique_time = int(time.time()) 
        self.save_name_pre = 't_{}'.format(unique_time)

        if not os.path.exists(training_args.output_dir):
            os.mkdir(training_args.output_dir)
        if not os.path.exists(training_args.output_dir+ "/tracking"):
            os.mkdir(training_args.output_dir+ "/tracking")
        if not os.path.exists(training_args.output_dir + "/" + model_args.tensorboard_path):
            os.mkdir(training_args.output_dir + "/" + model_args.tensorboard_path)

        # Tensorboard
        tensorboard_save_path = '{}/{}.pth'.format(training_args.output_dir + "/" + model_args.tensorboard_path, self.save_name_pre)
        writer = SummaryWriter(log_dir=tensorboard_save_path)

        return writer

    def track_values(self, train_loss, train_cov_loss, train_sim_loss, train_cross_entropy, train_cos_sim, cls_output):
        self.last_n_cov_losses.append(train_cov_loss)
        self.last_n_sim_losses.append(train_sim_loss)
        self.last_n_losses.append(train_loss)
        self.last_n_cross_entropy_losses.append(train_cross_entropy)
        self.last_n_cos_sim_losses.append(train_cos_sim)
        self.last_n_cls_outputs.append(cls_output)
        self.called_times += 1

    def save_values(self, writer, loss, train_cov_loss_weight, train_sim_loss_weight, weight_vector, bias_vector):
        def save_eigs(self, loss):
            R_eigs = loss.save_eigs() 
            R_eigs_df = pd.DataFrame(data=R_eigs, index=range(0, (self.called_times // self.save_frequency)+1))
            R_eigs_df.to_csv('{}/{}_R_eigs.csv'.format(self.output_dir, self.save_name_pre), index_label='called_times')
            return R_eigs

        def update_results(self, train_loss, train_cov_loss, train_sim_loss, train_cov_loss_weight, train_sim_loss_weight, train_cross_entropy, train_cos_sim, cls_output_max, cls_output_mean, cls_output_std):
            self.results['train_loss'].append(train_loss)
            self.results['train_cov_loss'].append(train_cov_loss)
            self.results['train_sim_loss'].append(train_sim_loss)
            self.results['train_weighted_cov_loss'].append(train_cov_loss*train_cov_loss_weight)
            self.results['train_weighted_sim_loss'].append(train_sim_loss*train_sim_loss_weight)
            self.results['train_cross_entropy'].append(train_cross_entropy)
            self.results['train_cos_sim'].append(train_cos_sim)
            self.results['cls_output_max'].append(np.mean(cls_output_max))
            self.results['cls_output_mean'].append(np.mean(cls_output_mean))
            self.results['cls_output_std'].append(np.mean(cls_output_std))

        def update_tensorboard(self, writer, train_loss, train_cov_loss, train_sim_loss, train_cov_loss_weight, train_sim_loss_weight, train_cross_entropy, train_cos_sim, cls_output_max, cls_output_mean, cls_output_std):
            writer.add_scalar("Train/Loss", train_loss, (self.called_times // self.save_frequency))
            writer.add_scalar("Train/Cov_Loss", train_cov_loss, (self.called_times // self.save_frequency))
            writer.add_scalar("Train/Sim_Loss", train_sim_loss, (self.called_times // self.save_frequency))
            writer.add_scalar("Train/CL_Cross_Entropy_Sim", train_cross_entropy, (self.called_times // self.save_frequency))
            writer.add_scalar("Train/Weighted_Cov_Loss", train_cov_loss * train_cov_loss_weight, (self.called_times // self.save_frequency))
            writer.add_scalar("Train/Weighted_Sim_Loss", train_sim_loss * train_sim_loss_weight, (self.called_times // self.save_frequency))
            writer.add_scalar("Train/Cos_Sim", train_cos_sim, (self.called_times // self.save_frequency))
            writer.add_scalar("Train/CLS_Output_Max", np.mean(cls_output_max), (self.called_times // self.save_frequency))
            writer.add_scalar("Train/CLS_Output_Mean", np.mean(cls_output_mean), (self.called_times // self.save_frequency))
            writer.add_scalar("Train/CLS_Output_STD", np.mean(cls_output_std), (self.called_times // self.save_frequency))
            #writer.add_scalar("Train/low_R_eig", (R_eigs[(self.called_times // self.save_frequency),:] < 1e-8).sum(), (self.called_times // self.save_frequency))
            #writer.add_scalar("Train/high_R_eig", (R_eigs[(self.called_times // self.save_frequency),:] > 1e-1).sum(), (self.called_times // self.save_frequency))

        def save_stats(self):
            # Save list as csv
            data_frame = pd.DataFrame(data=self.results, index=range(1, (self.called_times // self.save_frequency)+1, 1))
            data_frame.to_csv('{}/{}_statistics.csv'.format(self.output_dir, self.save_name_pre), index_label='called_times')

        def save_vectors(self, weight_vector, bias_vector, max_vector, mean_vector, std_vector):
            # Save vectors
            if not os.path.exists("{}/vectors/".format(self.output_dir)):
                os.mkdir("{}/vectors/".format(self.output_dir))
            np.save('{}/vectors/{}_vectors_t_{}'.format(self.output_dir, self.save_name_pre, (self.called_times // self.save_frequency)+1), np.vstack((weight_vector, bias_vector, max_vector, mean_vector, std_vector)))

        if ((self.called_times % self.save_frequency) == 0):
            stacked = np.vstack(self.last_n_cls_outputs)
            weight_vector = weight_vector.cpu().detach().numpy()
            bias_vector = bias_vector.cpu().detach().numpy()
            max_vector = np.max(np.abs(stacked), axis=0)
            mean_vector = np.mean(stacked, axis=0)
            std_vector = np.std(stacked, axis=0)
            #R_eigs = save_eigs(self, loss)
            update_results(self, np.mean(self.last_n_losses), np.mean(self.last_n_cov_losses), np.mean(self.last_n_sim_losses), train_cov_loss_weight, train_sim_loss_weight, np.mean(self.last_n_cross_entropy_losses), np.mean(self.last_n_cos_sim_losses), max_vector, mean_vector, std_vector)
            update_tensorboard(self, writer, np.mean(self.last_n_losses), np.mean(self.last_n_cov_losses), np.mean(self.last_n_sim_losses), train_cov_loss_weight, train_sim_loss_weight, np.mean(self.last_n_cross_entropy_losses), np.mean(self.last_n_cos_sim_losses), max_vector, mean_vector, std_vector)
            save_stats(self)
            save_vectors(self, weight_vector, bias_vector, max_vector, mean_vector, std_vector)
            self.last_n_cov_losses = []
            self.last_n_sim_losses = []
            self.last_n_losses = []
            self.last_n_cross_entropy_losses = []
            self.last_n_cos_sim_losses = []
            self.last_n_cls_outputs = []
            writer.flush()
