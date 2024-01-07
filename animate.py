import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import animation
import matplotlib.patches as patches

import pandas as pd
import torch
import pickle
from scipy.spatial.distance import cdist

from design import color_dict


''' 
Class to animate play with predictions

https://matplotlib.org/stable/users/explain/animations/animations.html
https://matplotlib.org/stable/users/explain/colors/colors.html
https://jakevdp.github.io/blog/2012/08/18/matplotlib-animation-tutorial/
'''
class AnimatePreds():
    
    # initialize variables we need. 
    # Example: Need to initialize the figure that the animation command will return
    ''' 
    ngs_play : from ngs data, filtered to only have frames for one play from one game
    model_dir : directory with model info. Needs to have predictions folder.
    '''
    def __init__(self, ngs_play, model_dir: str, displayNumbers=False) -> None:
        
        data_dir = f"seq_clipped_sorted_data"
        self.title_fontsize = 10

        self.MAX_FIELD_PLAYERS = 22
        self.start_x = 0
        self.stop_x = 120
        self.start_y = 53.3
        self.ngs_play = ngs_play
        try:
            self.plays_df = pd.read_csv("./data/plays.csv")
        except:
            self.plays_df = pd.read_csv("../data/plays.csv")

        self.displayNumbers = displayNumbers
        k=3.5
        self.FIG_HEIGHT = k*1.1
        self.FIG_WIDTH = k*2.5

        # for predictions
        self.game_id = self.ngs_play.gameId.values[0]
        self.play_id = self.ngs_play.playId.values[0]
        full_id = f"{str(self.game_id)}_{str(self.play_id)}"

        self.preds = torch.load(f"/Users/scottmaran/Code/Python/bdb_2024/saved_models/{model_dir}/predictions/{full_id}.pt").detach().cpu()  # (seq_length, 12)
        with open(f'/Users/scottmaran/Code/Python/bdb_2024/cleaned_data/{data_dir}/play_info_dict/{full_id}', 'rb') as file:
            self.play_info_dict = pickle.load(file)
        self.player_ids = np.load(f'/Users/scottmaran/Code/Python/bdb_2024/cleaned_data/{data_dir}/player_ids/{full_id}.npy')
        self.def_ids = self.player_ids[0:12].astype(float)  # includes football id 

        self.context = torch.load(f"/Users/scottmaran/Code/Python/bdb_2024/cleaned_data/{data_dir}/context_vector/{full_id}.pt").detach().cpu()  # (seq_length, 12)
        print(f"context  ={self.context[0,0]}")
        self.line_of_scrimmage = self.context[0,0]
        self.yards_to_go = self.context[0,1]

        self.play_length = self.play_info_dict['final_frame_in_play']

        self.frames_list = np.arange(1, self.play_length+1)
        def_id_info = pd.merge(pd.DataFrame(self.def_ids, columns=['nflId']), self.ngs_play.loc[:,['nflId', 'displayName','jerseyNumber']], how='left', on='nflId').drop_duplicates()
        #temp_df = self.ngs_play.query("nflId in @self.def_ids").copy()[['displayName', 'jerseyNumber']].drop_duplicates().sort_index(ascending=True)
        
        self.display_names = def_id_info.displayName.values
        self.display_names[0] = 'Football'
        self.jersey_numbers = def_id_info.jerseyNumber.values
        self.display_names[0] = '-1'

        # pandas series of all play info
        self.play_info = self.plays_df.loc[(self.plays_df.gameId == self.game_id) & (self.plays_df.playId == self.play_id)].iloc[0]
        
        fig, (ax,ax2) = plt.subplots(1, 2, figsize=(self.FIG_WIDTH, self.FIG_HEIGHT), gridspec_kw={'width_ratios': [2,1]})
        self.fig = fig
        self.field_ax = ax
        self.pred_ax = ax2
        
        self.fig.suptitle(f"{self.play_info.playDescription}", fontsize=self.title_fontsize)
        # create new axis for home, away, jersey
        self.ax_home = self.field_ax.twinx()
        self.ax_away = self.field_ax.twinx()
        self.ax_football = self.field_ax.twinx()
        self.ax_tackler = self.field_ax.twinx()
        self.ax_jersey = self.field_ax.twinx()

        self.ax_tackler.set_zorder(3)
        self.ax_football.set_zorder(3)
        self.ax_jersey.set_zorder(3)
        self.ax_home.set_zorder(2)
        self.ax_away.set_zorder(1)

        # create new axis for predictions
        #self.ax_preds = self.pred_ax.twinx()
        
        self.ani = animation.FuncAnimation(self.fig, self.update, frames=len(self.frames_list),
                                          init_func=self.setup_plot, blit=False)
        
        plt.close()
        
    # initialization function for animation call
    def setup_plot(self):

        self.field_ax.set_title(f"Play Vizualization", fontsize=self.title_fontsize*.9)#, fontsize=self.FIG_WIDTH)
        self.pred_ax.set_title(f"Tackle Predictions", fontsize=self.title_fontsize*.9)
        #self.pred_ax.yaxis.set_visible(False)

        endzones = True
        linenumbers = True
        
        # set axis limits for ax
        self.set_axis_plots(self.field_ax, self.stop_x, self.start_y)
        self.set_axis_plots(self.ax_home, self.stop_x, self.start_y)
        self.set_axis_plots(self.ax_away, self.stop_x, self.start_y)
        self.set_axis_plots(self.ax_football, self.stop_x, self.start_y)
        self.set_axis_plots(self.ax_tackler, self.stop_x, self.start_y)
        self.set_axis_plots(self.ax_jersey, self.stop_x, self.start_y)

        # set axis limits for ax2
        self.pred_ax.set_xlim([0,1])
        #self.ax_preds.yaxis.set_visible(False)

        # set up colors and patches for field
        self.set_up_field(endzones, linenumbers)#, line_of_scrimmage=self.line_of_scrimmage, yards_to_first=self.yards_to_go)

        # create scatterplots on axis for data
        self.scat_field = self.field_ax.scatter([], [], s=90, color='orange',edgecolors='black')
        self.scat_home = self.ax_home.scatter([], [], s=90, color='xkcd:aquamarine',edgecolors='black')
        self.scat_away = self.ax_away.scatter([], [], s=90, color='xkcd:salmon',edgecolors='black')
        self.scat_football = self.ax_football.scatter([], [], s=90, color='orange',edgecolors='black', label='Football')
        self.scat_tackler = self.ax_tackler.scatter([], [], s=90, color='yellow',edgecolors='black', label='True Tackler')
        # self.scat_field = self.field_ax.scatter([], [], s=90, color=np.array(color_dict['green_rgb']).reshape(1,-2),edgecolors='black')
        # self.scat_home = self.ax_home.scatter([], [], s=90, color=np.array(color_dict['blue_rgb']).reshape(1,-2),edgecolors='black')
        # self.scat_away = self.ax_away.scatter([], [], s=90, color=np.array(color_dict['red_rgb']).reshape(1,-2),edgecolors='black')
        # self.scat_football = self.ax_football.scatter([], [], s=90, color=np.array(color_dict['brown_rgb']).reshape(1,-2),edgecolors='black', label='Football')
        # self.scat_tackler = self.ax_tackler.scatter([], [], s=90, color=np.array(color_dict['yellow_rgb']).reshape(1,-2),edgecolors='black', label='True Tackler')
        

        # get play summary
        self.tackler_id = self.play_info_dict['tackler_id']

        # add direction stats and jersey numbers/names
        self._scat_jersey_list = []
        self._scat_number_list = []
        self._scat_name_list = []
        self._a_dir_list = []
        self._a_or_list = []
        for _ in range(self.MAX_FIELD_PLAYERS):
            self._scat_jersey_list.append(self.ax_jersey.text(0, 0, '', horizontalalignment = 'center', verticalalignment = 'center', c = 'white'))
            self._scat_number_list.append(self.ax_jersey.text(0, 0, '', horizontalalignment = 'center', verticalalignment = 'center', c = 'black', fontsize=6))
            self._scat_name_list.append(self.ax_jersey.text(0, 0, '', horizontalalignment = 'center', verticalalignment = 'center', c = 'black'))
            
            self._a_dir_list.append(self.field_ax.add_patch(patches.Arrow(0, 0, 0, 0, color = 'k')))
            self._a_or_list.append(self.field_ax.add_patch(patches.Arrow(0, 0, 0, 0, color = 'k')))

        ''' set up second axis '''

        # change def ids to display names
        BAR_DISPLAY_LABELS = self.jersey_numbers 
        BAR_DISPLAY_LABELS = BAR_DISPLAY_LABELS.astype(int)
        BAR_DISPLAY_LABELS = list(BAR_DISPLAY_LABELS)
        BAR_DISPLAY_LABELS[0] = 'No \n Tackle'
        print(f"display names = {BAR_DISPLAY_LABELS}")
        self.bar_preds = self.pred_ax.barh(y=np.arange(len(self.def_ids)), width=np.zeros(len(self.def_ids)), tick_label=BAR_DISPLAY_LABELS,color='xkcd:turquoise')
        self._bar_preds_list = []
        for index, rectangle in enumerate(self.bar_preds):
            if self.def_ids[index] == self.tackler_id:
                rectangle.set_color('y')
            self._bar_preds_list.append(rectangle)

        self.pred_ax.set_xlabel('Probability (%)')
        self.pred_ax.set_xticks(np.arange(0,110,25).astype(int))

        self.pred_ax_names = self.pred_ax.twinx()
        print(self.display_names)
        NAMES = [name.split(" ")[-1] for name in self.display_names]
        self.pred_ax_names.barh(y=np.arange(len(self.def_ids)), width=np.zeros(len(self.def_ids)), tick_label=NAMES,color='xkcd:turquoise')

        # return all axis plots that you want to update
        return (self.scat_field, self.scat_home, self.scat_away, self.scat_football, self.scat_tackler, self.bar_preds, *self._scat_jersey_list, *self._scat_number_list, *self._scat_name_list)
    
    def update(self, i):

        frame = self.frames_list[i] # 1 to last_frame
        time_df = self.ngs_play.query("frameId == @frame")
        
        (label1, label2, label3) = time_df.club.unique()    #football is last label

        self.scat_home.set_offsets(time_df.loc[time_df.club == label2, ['x','y']].to_numpy())
        self.scat_away.set_offsets(time_df.loc[time_df.club == label1, ['x','y']].to_numpy())
        self.scat_tackler.set_offsets(time_df.loc[time_df.nflId == self.tackler_id, ['x','y']].to_numpy())
        self.scat_football.set_offsets(time_df.loc[time_df.club == label3, ['x','y']].to_numpy())
        #self.scat_field.set_offsets(time_df.loc[time_df.club == label3, ['x','y']].to_numpy())

        self.ax_tackler.legend(loc= (0,0), framealpha=0.7)

        # 2nd axis
        # for index in range(len(self.bar_preds)):
        #     self.bar_preds[index].set_width(self.preds[frame-1][index])
        for ind, rectangle in enumerate(self._bar_preds_list):
            rectangle.set_width(self.preds[frame-1][ind]*100)

        
        #add direction and jersey info
        #jersey_df = time_df[time_df.nflId != 0].reset_index()
        jersey_df = time_df[time_df.displayName != 'football'].reset_index()

        for (index, row) in jersey_df.iterrows():
            #self._scat_jersey_list[index].set_position((row.x, row.y))
            #self._scat_jersey_list[index].set_text(row.position)
            
            if self.displayNumbers and (row.jerseyNumber in self.jersey_numbers):
                self._scat_number_list[index].set_position((row.x, row.y))
                self._scat_number_list[index].set_text(int(row.jerseyNumber))
            #self._scat_name_list[index].set_position((row.x, row.y-1.9))
            #self._scat_name_list[index].set_text(row.displayName.split()[-1])
            
            player_orientation_rad = self.deg_to_rad(self.convert_orientation(row.o))
            player_direction_rad = self.deg_to_rad(self.convert_orientation(row.dir))
            player_speed = row.s
            
            player_vel = np.array([np.real(self.polar_to_z(player_speed, player_direction_rad)), np.imag(self.polar_to_z(player_speed, player_direction_rad))])
            player_orient = np.array([np.real(self.polar_to_z(2, player_orientation_rad)), np.imag(self.polar_to_z(2, player_orientation_rad))])
            
            self._a_dir_list[index].remove()
            self._a_dir_list[index] = self.field_ax.add_patch(patches.Arrow(row.x, row.y, player_vel[0], player_vel[1], color = 'k'))
            
            self._a_or_list[index].remove()
            self._a_or_list[index] = self.field_ax.add_patch(patches.Arrow(row.x, row.y, player_orient[0], player_orient[1], color = 'grey', width = 2))

        if i == self.play_length-1:
            self.field_ax.clear()
            self.ax_home.clear()
            self.ax_away.clear()
            self.ax_football.clear()
            self.ax_tackler.clear()
            self.ax_jersey.clear()
            self.pred_ax.clear()

        return (self.scat_field, self.scat_home, self.scat_away, self.scat_tackler, self.bar_preds, *self._scat_jersey_list, *self._scat_number_list, *self._scat_name_list)
    
    def set_up_field(self, endzones=True, linenumbers=True, line_of_scrimmage=-1, yards_to_first=-1) -> None:
        yard_numbers_size = self.fig.get_size_inches()[0]*1.5

        # color field 
        rect = patches.Rectangle((0, 0), 120, 53.3, linewidth=0.1,
                                    edgecolor='r', facecolor='white', zorder=0)
        self.field_ax.add_patch(rect)

        # plot
        self.field_ax.plot([10, 10, 10, 20, 20, 30, 30, 40, 40, 50, 50, 60, 60, 70, 70, 80,
                    80, 90, 90, 100, 100, 110, 110, 120, 0, 0, 120, 120],
                    [0, 0, 53.3, 53.3, 0, 0, 53.3, 53.3, 0, 0, 53.3, 53.3, 0, 0, 53.3,
                    53.3, 0, 0, 53.3, 53.3, 0, 0, 53.3, 53.3, 53.3, 0, 0, 53.3],
                    color='grey')
        
        if line_of_scrimmage > 0:
            los = patches.Rectangle((10+line_of_scrimmage, 0), 10+line_of_scrimmage+1, 53.3,
                                    linewidth=0.01,
                                    edgecolor='r',
                                    facecolor='blue',
                                    alpha=0.2,
                                    zorder=0)
            self.field_ax.add_patch(los)
            # need line of scrimmage to use yards to first
            if yards_to_first > 0:
                first = patches.Rectangle((10+line_of_scrimmage+yards_to_first, 0), 10+line_of_scrimmage+yards_to_first+1, 53.3,
                                        linewidth=0.01,
                                        edgecolor='black',
                                        facecolor='yellow',
                                        alpha=0.2,
                                        zorder=0)
                self.field_ax.add_patch(first)

        # Endzones
        if endzones:
            ez1 = patches.Rectangle((0, 0), 10, 53.3,
                                    linewidth=0.1,
                                    edgecolor='r',
                                    facecolor='green',
                                    alpha=0.2,
                                    zorder=0)
            ez2 = patches.Rectangle((110, 0), 120, 53.3,
                                    linewidth=0.1,
                                    edgecolor='r',
                                    facecolor='green',
                                    alpha=0.2,
                                    zorder=0)
            self.field_ax.add_patch(ez1)
            self.field_ax.add_patch(ez2)
            
        if endzones:
            hash_range = range(11, 110)
        else:
            hash_range = range(1, 120)

        # add hashes
        for x in hash_range:
            self.field_ax.plot([x, x], [0.4, 0.7], color='grey')
            self.field_ax.plot([x, x], [53.0, 52.5], color='grey')
            self.field_ax.plot([x, x], [22.91, 23.57], color='grey')
            self.field_ax.plot([x, x], [29.73, 30.39], color='grey')
            
        # add linenumbers
        if linenumbers:
                for x in range(20, 110, 10):
                    numb = x
                    if x > 50:
                        numb = 120 - x
                    self.field_ax.text(x, 5, str(numb - 10),
                            horizontalalignment='center',
                            fontsize=yard_numbers_size,  # fontname='Arial',
                            color='grey')
                    self.field_ax.text(x - 0.95, 53.3 - 5, str(numb - 10),
                            horizontalalignment='center',
                            fontsize=yard_numbers_size,  # fontname='Arial',
                            color='grey', rotation=180)

        self.field_ax.set_xlim(self.start_x, self.stop_x)
        self.field_ax.set_ylim(0, self.start_y)
        self.field_ax.set_xticks(range(self.start_x,self.stop_x, 10))

    @staticmethod
    def set_axis_plots(ax, max_x, max_y) -> None:
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)

        ax.set_xlim([0, max_x])
        ax.set_ylim([0, max_y])
        
    @staticmethod
    def convert_orientation(x):
        return (-x + 90)%360
    
    @staticmethod
    def polar_to_z(r, theta):
        return r * np.exp( 1j * theta)
    
    @staticmethod
    def deg_to_rad(deg):
        return deg*np.pi/180
    


# seq : shape (seq_length, 23,11)
def get_closest_defender(seq):
    ball_and_opps = seq[:,0:12, 3:5]
    dists = torch.cdist(ball_and_opps[:,0:1,:], ball_and_opps, p=2.0)
    dists[:,0,0] = torch.inf 
    closest_defender = dists.argmin(dim=2).squeeze(-1)
    return closest_defender

''' 
Class to animate plays (only relies on given data)
'''
class AnimateClosestTackler():
    
    # initialize variables we need. 
    # Example: Need to initialize the figure that the animation command will return
    ''' 
    ngs_play : from ngs data, filtered to only have frames for one play from one game
    data_dir : directory where cleaned data is
    '''
    def __init__(self, ngs_play, data_dir='seq_clipped_sorted_data', displayNumbers=False) -> None:
        
        self.MAX_FIELD_PLAYERS = 22
        self.start_x = 0
        self.stop_x = 120
        self.start_y = 53.3

        # make football id -1
        self.ngs_play = ngs_play
        self.ngs_play.loc[self.ngs_play.displayName == 'football', 'nflId'] = -1

        self.plays_df = pd.read_csv("../data/plays.csv")

        self.displayNumbers = displayNumbers
        self.FIG_HEIGHT = 4
        self.FIG_WIDTH = 8

        # for predictions
        self.game_id = self.ngs_play.gameId.values[0]
        self.play_id = self.ngs_play.playId.values[0]
        full_id = f"{str(self.game_id)}_{str(self.play_id)}"

        with open(f'/Users/scottmaran/Code/Python/bdb_2024/cleaned_data/{data_dir}/play_info_dict/{full_id}', 'rb') as file:
            self.play_info_dict = pickle.load(file)
        self.player_ids = np.load(f'/Users/scottmaran/Code/Python/bdb_2024/cleaned_data/{data_dir}/player_ids/{full_id}.npy')
        self.def_ids = self.player_ids[0:12].astype(float)  # includes football id 

        print(f"self.def_ids = {self.def_ids}, shape = {self.def_ids.shape}")

        def_id_info = pd.merge(pd.DataFrame(self.def_ids, columns=['nflId']), self.ngs_play.loc[:,['nflId', 'displayName','jerseyNumber']], how='left', on='nflId').drop_duplicates()
        #temp_df = self.ngs_play.query("nflId in @self.def_ids").copy()[['displayName', 'jerseyNumber']].drop_duplicates().sort_index(ascending=True)
        self.display_names = def_id_info.displayName.values
        self.jersey_numbers = def_id_info.jerseyNumber.values

        play_length = self.play_info_dict['final_frame_in_play']
        self.frames_list = np.arange(1, play_length+1)
        print(self.frames_list)

        # pandas series of all play info
        self.play_info = self.plays_df.loc[(self.plays_df.gameId == self.game_id) & (self.plays_df.playId == self.play_id)].iloc[0]
        
        fig, ax = plt.subplots(1, 1, figsize=(self.FIG_WIDTH, self.FIG_HEIGHT))
        self.fig = fig
        self.field_ax = ax
        # create new axis for home, away, jersey
        self.ax_home = self.field_ax.twinx()
        self.ax_away = self.field_ax.twinx()
        self.ax_jersey = self.field_ax.twinx()
        self.ax_summary = self.field_ax.twinx()
        self.closest_player = self.field_ax.twinx()
        
        self.ani = animation.FuncAnimation(self.fig, self.update, frames=len(self.frames_list),
                                          init_func=self.setup_plot, blit=False)
        
        plt.close()
        
    # initialization function for animation call
    def setup_plot(self):

        self.field_ax.set_title(f"{self.play_info.playDescription}", fontsize=self.FIG_WIDTH)

        endzones = True
        linenumbers = True
        
        # set axis limits for ax
        self.set_axis_plots(self.field_ax, self.stop_x, self.start_y)
        self.set_axis_plots(self.ax_home, self.stop_x, self.start_y)
        self.set_axis_plots(self.ax_away, self.stop_x, self.start_y)
        self.set_axis_plots(self.ax_jersey, self.stop_x, self.start_y)
        self.set_axis_plots(self.ax_summary, self.stop_x, self.start_y)
        self.set_axis_plots(self.closest_player, self.stop_x, self.start_y)


        # set up colors and patches for field
        self.set_up_field(endzones, linenumbers)
        
        # create scatterplots on axis for data
        self.scat_field = self.field_ax.scatter([], [], s=90, color='orange',edgecolors='black')
        self.scat_home = self.ax_home.scatter([], [], s=90, color='xkcd:aquamarine',edgecolors='black')
        self.scat_away = self.ax_away.scatter([], [], s=90, color='xkcd:salmon',edgecolors='black')
        
        # get play summary
        tackler_id = self.play_info_dict['tackler_id']

        self.scat_summary = self.ax_summary.text(0, 0, f"True Tackler={self.jersey_numbers[int(np.where(self.def_ids == tackler_id)[0])]}", fontsize = self.FIG_WIDTH, 
                                           bbox=dict(boxstyle="square", facecolor="white"),
                                           horizontalalignment = 'left', verticalalignment = 'top', c = 'black')
        self.scat_closest_player = self.closest_player.text(0, 0, f"Closest Tackler=0", fontsize = self.FIG_WIDTH, 
                                           bbox=dict(boxstyle="square", facecolor="white"),
                                           horizontalalignment = 'right', verticalalignment = 'top', c = 'black')
        
        # add direction stats and jersey numbers/names
        self._scat_jersey_list = []
        self._scat_number_list = []
        self._scat_name_list = []
        self._a_dir_list = []
        self._a_or_list = []
        for _ in range(self.MAX_FIELD_PLAYERS):
            self._scat_jersey_list.append(self.ax_jersey.text(0, 0, '', horizontalalignment = 'center', verticalalignment = 'center', c = 'white'))
            self._scat_number_list.append(self.ax_jersey.text(0, 0, '', horizontalalignment = 'center', verticalalignment = 'center', c = 'black', fontsize=6))
            self._scat_name_list.append(self.ax_jersey.text(0, 0, '', horizontalalignment = 'center', verticalalignment = 'center', c = 'black'))
            
            self._a_dir_list.append(self.field_ax.add_patch(patches.Arrow(0, 0, 0, 0, color = 'k')))
            self._a_or_list.append(self.field_ax.add_patch(patches.Arrow(0, 0, 0, 0, color = 'k')))

        plt.show()
        # return all axis plots that you want to update
        return (self.scat_field, self.scat_home, self.scat_away, self.scat_summary, self.scat_closest_player, *self._scat_jersey_list, *self._scat_number_list, *self._scat_name_list)
    
    def get_closest_tackler(self, time_df):
        time_df.loc[time_df.displayName == 'Football', 'nflId'] = -1

    def update(self, i):
        frame = self.frames_list[i] # 1 to last_frame
        time_df = self.ngs_play.query("frameId == @frame")
        
        (label1, label2, label3) = time_df.club.unique()    #football is last label

        self.scat_field.set_offsets(time_df.loc[time_df.club == label3, ['x','y']].to_numpy())
        self.scat_home.set_offsets(time_df.loc[time_df.club == label2, ['x','y']].to_numpy())
        self.scat_away.set_offsets(time_df.loc[time_df.club == label1, ['x','y']].to_numpy())

        # get closest player to football

        # print(f'time ids = {time_df.nflId.values}')
        def_df = time_df.query("nflId in @self.def_ids").sort_values('nflId')[['x', 'y']].values # (12, 2)
        # print(f"def_df shape = {def_df.shape}")
        dists = cdist(def_df[0:1,:], def_df, metric='euclidean')                   # (1, 12)
        # print(dists)
        # print(f"dists shape = {dists.shape}")
        dists[0,0] = np.inf 
        # print(dists)
        closest_defender_index = np.argmin(dists, axis=-1).squeeze(-1)
        # print(closest_defender_index)
        # print(f"closest_defender shape = {closest_defender_index.shape}")
        closest_defender = self.jersey_numbers[closest_defender_index]
        self.scat_closest_player.set_text(f'Closest Tackler={closest_defender}')

        

        jersey_df = time_df[time_df.displayName != 'football'].reset_index()

        for (index, row) in jersey_df.iterrows():
            #self._scat_jersey_list[index].set_position((row.x, row.y))
            #self._scat_jersey_list[index].set_text(row.position)
            
            if self.displayNumbers and (row.jerseyNumber in self.jersey_numbers):
                self._scat_number_list[index].set_position((row.x, row.y))
                self._scat_number_list[index].set_text(int(row.jerseyNumber))
            #self._scat_name_list[index].set_position((row.x, row.y-1.9))
            #self._scat_name_list[index].set_text(row.displayName.split()[-1])
            
            player_orientation_rad = self.deg_to_rad(self.convert_orientation(row.o))
            player_direction_rad = self.deg_to_rad(self.convert_orientation(row.dir))
            player_speed = row.s
            
            player_vel = np.array([np.real(self.polar_to_z(player_speed, player_direction_rad)), np.imag(self.polar_to_z(player_speed, player_direction_rad))])
            player_orient = np.array([np.real(self.polar_to_z(2, player_orientation_rad)), np.imag(self.polar_to_z(2, player_orientation_rad))])
            
            self._a_dir_list[index].remove()
            self._a_dir_list[index] = self.field_ax.add_patch(patches.Arrow(row.x, row.y, player_vel[0], player_vel[1], color = 'k'))
            
            self._a_or_list[index].remove()
            self._a_or_list[index] = self.field_ax.add_patch(patches.Arrow(row.x, row.y, player_orient[0], player_orient[1], color = 'grey', width = 2))
                

        return (self.scat_field, self.scat_home, self.scat_away, self.scat_summary, self.scat_closest_player, *self._scat_jersey_list, *self._scat_number_list, *self._scat_name_list)
    
    def set_up_field(self, endzones=True, linenumbers=True, line_of_scrimmage=-1, yards_to_first=-1) -> None:
        yard_numbers_size = self.fig.get_size_inches()[0]*1.5

        # color field 
        rect = patches.Rectangle((0, 0), 120, 53.3, linewidth=0.1,
                                    edgecolor='r', facecolor='white', zorder=0)
        self.field_ax.add_patch(rect)

        # plot
        self.field_ax.plot([10, 10, 10, 20, 20, 30, 30, 40, 40, 50, 50, 60, 60, 70, 70, 80,
                    80, 90, 90, 100, 100, 110, 110, 120, 0, 0, 120, 120],
                    [0, 0, 53.3, 53.3, 0, 0, 53.3, 53.3, 0, 0, 53.3, 53.3, 0, 0, 53.3,
                    53.3, 0, 0, 53.3, 53.3, 0, 0, 53.3, 53.3, 53.3, 0, 0, 53.3],
                    color='black')
        
        if line_of_scrimmage > 0:
            los = patches.Rectangle((10+line_of_scrimmage, 0), 10+line_of_scrimmage+1, 53.3,
                                    linewidth=0.1,
                                    edgecolor='r',
                                    facecolor='blue',
                                    alpha=0.2,
                                    zorder=0)
            self.field_ax.add_patch(los)
            # need line of scrimmage to use yards to first
            if yards_to_first > 0:
                first = patches.Rectangle((10+line_of_scrimmage+yards_to_first, 0), 10+line_of_scrimmage+yards_to_first+1, 53.3,
                                        linewidth=0.1,
                                        edgecolor='black',
                                        facecolor='yellow',
                                        alpha=0.2,
                                        zorder=0)
                self.field_ax.add_patch(first)

        # Endzones
        if endzones:
            ez1 = patches.Rectangle((0, 0), 10, 53.3,
                                    linewidth=0.1,
                                    edgecolor='r',
                                    facecolor='green',
                                    alpha=0.2,
                                    zorder=0)
            ez2 = patches.Rectangle((110, 0), 120, 53.3,
                                    linewidth=0.1,
                                    edgecolor='r',
                                    facecolor='green',
                                    alpha=0.2,
                                    zorder=0)
            self.field_ax.add_patch(ez1)
            self.field_ax.add_patch(ez2)
            
        if endzones:
            hash_range = range(11, 110)
        else:
            hash_range = range(1, 120)

        # add hashes
        for x in hash_range:
            self.field_ax.plot([x, x], [0.4, 0.7], color='black')
            self.field_ax.plot([x, x], [53.0, 52.5], color='black')
            self.field_ax.plot([x, x], [22.91, 23.57], color='black')
            self.field_ax.plot([x, x], [29.73, 30.39], color='black')
            
        # add linenumbers
        if linenumbers:
                for x in range(20, 110, 10):
                    numb = x
                    if x > 50:
                        numb = 120 - x
                    self.field_ax.text(x, 5, str(numb - 10),
                            horizontalalignment='center',
                            fontsize=yard_numbers_size,  # fontname='Arial',
                            color='black')
                    self.field_ax.text(x - 0.95, 53.3 - 5, str(numb - 10),
                            horizontalalignment='center',
                            fontsize=yard_numbers_size,  # fontname='Arial',
                            color='black', rotation=180)

        self.field_ax.set_xlim(self.start_x, self.stop_x)
        self.field_ax.set_ylim(0, self.start_y)
        self.field_ax.set_xticks(range(self.start_x,self.stop_x, 10))

    @staticmethod
    def set_axis_plots(ax, max_x, max_y) -> None:
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)

        ax.set_xlim([0, max_x])
        ax.set_ylim([0, max_y])
        
    @staticmethod
    def convert_orientation(x):
        return (-x + 90)%360
    
    @staticmethod
    def polar_to_z(r, theta):
        return r * np.exp( 1j * theta)
    
    @staticmethod
    def deg_to_rad(deg):
        return deg*np.pi/180
    
######################
######################
######################
######################
######################
######################
######################
######################
######################
######################




















































































''' 
Class to animate play with predictions
'''
class AnimateFeature():
    
    # initialize variables we need. 
    # Example: Need to initialize the figure that the animation command will return
    ''' 
    ngs_play : from ngs data, filtered to only have frames for one play from one game
    '''
    def __init__(self, ngs_play, displayNumbers=False, include_preds=False) -> None:
        
        self.MAX_FIELD_PLAYERS = 22
        self.start_x = 0
        self.stop_x = 120
        self.start_y = 53.3
        self.ngs_play = ngs_play
        self.frames_list = ngs_play.frameId.unique()

        self.include_preds = include_preds
        self.preds = None
        self.play_info = None

        self.displayNumbers = displayNumbers
        self.FIG_HEIGHT = 4
        self.FIG_WIDTH = 8

        if self.include_preds:
            play_id = f"{str(self.ngs_play.gameId.values[0])}_{str(self.ngs_play.playId.values[0])}"
            self.preds = torch.load(f"saved_models/lstm/predictions/{play_id}.pt").detach()  # (MAX_BATCH_SEQ_LEN, 12)
            self.play_info = np.load(f"cleaned_data/seq_data/full_seq_info/{play_id}.npy")
        
        fig, ax = plt.subplots(1, figsize=(self.FIG_WIDTH, self.FIG_HEIGHT))
        
        self.fig = fig
        self.field_ax = ax
        
        # create new axis for home, away, jersey
        self.ax_home = self.field_ax.twinx()
        self.ax_away = self.field_ax.twinx()
        self.ax_jersey = self.field_ax.twinx()
        self.ax_pred = self.field_ax.twinx()
        
        self.ani = animation.FuncAnimation(self.fig, self.update, frames=len(self.frames_list),
                                          init_func=self.setup_plot, blit=False)
        
        plt.close()
        
    # initialization function for animation call
    def setup_plot(self):

        endzones = True
        linenumbers = True
        
        # set axis limits
        self.set_axis_plots(self.field_ax, self.stop_x, self.start_y)
        self.set_axis_plots(self.ax_home, self.stop_x, self.start_y)
        self.set_axis_plots(self.ax_away, self.stop_x, self.start_y)
        self.set_axis_plots(self.ax_jersey, self.stop_x, self.start_y)
        self.set_axis_plots(self.ax_pred, self.stop_x, self.start_y)

        # set up colors and patches for field
        self.set_up_field(endzones, linenumbers)
        
        # create scatterplots on axis for data
        self.scat_field = self.field_ax.scatter([], [], s=50, color='orange')
        self.scat_home = self.ax_home.scatter([], [], s=50, color='blue')
        self.scat_away = self.ax_away.scatter([], [], s=50, color='red')
        
        # create box for prediction
        self.scat_pred = self.ax_pred.text(0, 0, '', fontsize = self.FIG_WIDTH, 
                                           bbox=dict(boxstyle="square", facecolor="white"),
                                           horizontalalignment = 'left', verticalalignment = 'top', c = 'black')
        
        # add direction stats and jersey numbers/names
        self._scat_jersey_list = []
        self._scat_number_list = []
        self._scat_name_list = []
        self._a_dir_list = []
        self._a_or_list = []
        for _ in range(self.MAX_FIELD_PLAYERS):
            self._scat_jersey_list.append(self.ax_jersey.text(0, 0, '', horizontalalignment = 'center', verticalalignment = 'center', c = 'white'))
            self._scat_number_list.append(self.ax_jersey.text(0, 0, '', horizontalalignment = 'center', verticalalignment = 'center', c = 'black'))
            self._scat_name_list.append(self.ax_jersey.text(0, 0, '', horizontalalignment = 'center', verticalalignment = 'center', c = 'black'))
            
            self._a_dir_list.append(self.field_ax.add_patch(patches.Arrow(0, 0, 0, 0, color = 'k')))
            self._a_or_list.append(self.field_ax.add_patch(patches.Arrow(0, 0, 0, 0, color = 'k')))
        
        # return all axis plots that you want to update
        return (self.scat_field, self.scat_home, self.scat_away, self.ax_pred, *self._scat_jersey_list, *self._scat_number_list, *self._scat_name_list)
    
    def update(self, i):
        frame = self.frames_list[i] # 1 to last_frame
        time_df = self.ngs_play.query("frameId == @frame")
        
        #time_df['team_indicator'] = self.add_team_indicator(time_df)
        #print(time_df)
        
        (label1, label2, label3) = time_df.club.unique()    #football is last label

        self.scat_field.set_offsets(time_df.loc[time_df.club == label3, ['x','y']].to_numpy())
        self.scat_home.set_offsets(time_df.loc[time_df.club == label2, ['x','y']].to_numpy())
        self.scat_away.set_offsets(time_df.loc[time_df.club == label1, ['x','y']].to_numpy())
    
        # add prediction
        if self.include_preds:

            player_id_pred_dict = dict(zip(self.play_info[-12:],self.preds[frame-1])) # e.g. {10111:0.14, ...}
            sorted_dict = dict(sorted(player_id_pred_dict.items(), key=lambda item: item[1], reverse=True))

            set_str = ""
            try:
                for id,prob in sorted_dict.items():
                    name, jersey_num = time_df.loc[time_df.nflId == id, ['displayName', 'jerseyNumber']].values[0]
                    set_str += f"P({name})={str(prob.item())[0:4]} ({jersey_num}) \n"
            except:
                pass
                #id not in
            self.scat_pred.set_text(set_str)
        
        #add direction and jersey info
        #jersey_df = time_df[time_df.nflId != 0].reset_index()
        jersey_df = time_df[time_df.displayName != 'football'].reset_index()

        for (index, row) in jersey_df.iterrows():
            #self._scat_jersey_list[index].set_position((row.x, row.y))
            #self._scat_jersey_list[index].set_text(row.position)
            
            if self.displayNumbers:
                self._scat_number_list[index].set_position((row.x, row.y + 1.9))
                self._scat_number_list[index].set_text(int(row.jerseyNumber))
            #self._scat_name_list[index].set_position((row.x, row.y-1.9))
            #self._scat_name_list[index].set_text(row.displayName.split()[-1])
            
            player_orientation_rad = self.deg_to_rad(self.convert_orientation(row.o))
            player_direction_rad = self.deg_to_rad(self.convert_orientation(row.dir))
            player_speed = row.s
            
            player_vel = np.array([np.real(self.polar_to_z(player_speed, player_direction_rad)), np.imag(self.polar_to_z(player_speed, player_direction_rad))])
            player_orient = np.array([np.real(self.polar_to_z(2, player_orientation_rad)), np.imag(self.polar_to_z(2, player_orientation_rad))])
            
            self._a_dir_list[index].remove()
            self._a_dir_list[index] = self.field_ax.add_patch(patches.Arrow(row.x, row.y, player_vel[0], player_vel[1], color = 'k'))
            
            self._a_or_list[index].remove()
            self._a_or_list[index] = self.field_ax.add_patch(patches.Arrow(row.x, row.y, player_orient[0], player_orient[1], color = 'grey', width = 2))
                

        return (self.scat_field, self.scat_home, self.scat_away, self.ax_pred, *self._scat_jersey_list, *self._scat_number_list, *self._scat_name_list)
    
    def set_up_field(self, endzones=True, linenumbers=True) -> None:
        yard_numbers_size = self.fig.get_size_inches()[0]*1.5

        # color field 
        rect = patches.Rectangle((0, 0), 120, 53.3, linewidth=0.1,
                                    edgecolor='r', facecolor='darkgreen', zorder=0)
        self.field_ax.add_patch(rect)

        # plot
        self.field_ax.plot([10, 10, 10, 20, 20, 30, 30, 40, 40, 50, 50, 60, 60, 70, 70, 80,
                    80, 90, 90, 100, 100, 110, 110, 120, 0, 0, 120, 120],
                    [0, 0, 53.3, 53.3, 0, 0, 53.3, 53.3, 0, 0, 53.3, 53.3, 0, 0, 53.3,
                    53.3, 0, 0, 53.3, 53.3, 0, 0, 53.3, 53.3, 53.3, 0, 0, 53.3],
                    color='white')

        # Endzones
        if endzones:
            ez1 = patches.Rectangle((0, 0), 10, 53.3,
                                    linewidth=0.1,
                                    edgecolor='r',
                                    facecolor='blue',
                                    alpha=0.2,
                                    zorder=0)
            ez2 = patches.Rectangle((110, 0), 120, 53.3,
                                    linewidth=0.1,
                                    edgecolor='r',
                                    facecolor='blue',
                                    alpha=0.2,
                                    zorder=0)
            self.field_ax.add_patch(ez1)
            self.field_ax.add_patch(ez2)
            
        if endzones:
            hash_range = range(11, 110)
        else:
            hash_range = range(1, 120)

        # add hashes
        for x in hash_range:
            self.field_ax.plot([x, x], [0.4, 0.7], color='white')
            self.field_ax.plot([x, x], [53.0, 52.5], color='white')
            self.field_ax.plot([x, x], [22.91, 23.57], color='white')
            self.field_ax.plot([x, x], [29.73, 30.39], color='white')
            
        # add linenumbers
        if linenumbers:
                for x in range(20, 110, 10):
                    numb = x
                    if x > 50:
                        numb = 120 - x
                    self.field_ax.text(x, 5, str(numb - 10),
                            horizontalalignment='center',
                            fontsize=yard_numbers_size,  # fontname='Arial',
                            color='white')
                    self.field_ax.text(x - 0.95, 53.3 - 5, str(numb - 10),
                            horizontalalignment='center',
                            fontsize=yard_numbers_size,  # fontname='Arial',
                            color='white', rotation=180)

        self.field_ax.set_xlim(self.start_x, self.stop_x)
        self.field_ax.set_ylim(0, self.start_y)
        self.field_ax.set_xticks(range(self.start_x,self.stop_x, 10))

    @staticmethod
    def set_axis_plots(ax, max_x, max_y) -> None:
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)

        ax.set_xlim([0, max_x])
        ax.set_ylim([0, max_y])
        
    @staticmethod
    def convert_orientation(x):
        return (-x + 90)%360
    
    @staticmethod
    def polar_to_z(r, theta):
        return r * np.exp( 1j * theta)
    
    @staticmethod
    def deg_to_rad(deg):
        return deg*np.pi/180
        


''' 
Class to animate NFL Plays from NGS data
'''
class AnimatePlay():
    
    # initialize variables we need. 
    # Example: Need to initialize the figure that the animation command will return
    def __init__(self, play_df) -> None:
        
        self.MAX_FIELD_PLAYERS = 22
        self.start_x = 0
        self.stop_x = 120
        self.start_y = 53.3
        self.play_df = play_df
        self.frames_list = play_df.frameId.unique()
        self.games_df = pd.read_csv("data/games.csv")
        
        fig, ax = plt.subplots(1, figsize=(8,4))
        
        self.fig = fig
        self.field_ax = ax
        
        # create new axis for home, away, jersey
        self.ax_home = self.field_ax.twinx()
        self.ax_away = self.field_ax.twinx()
        self.ax_jersey = self.field_ax.twinx()
        
        self.ani = animation.FuncAnimation(self.fig, self.update, frames=len(self.frames_list),
                                          init_func=self.setup_plot, blit=False)
        
        plt.close()
        
    # initialization function for animation call
    def setup_plot(self):

        endzones = True
        linenumbers = True
        
        # set axis limits
        self.set_axis_plots(self.field_ax, self.stop_x, self.start_y)
        self.set_axis_plots(self.ax_home, self.stop_x, self.start_y)
        self.set_axis_plots(self.ax_away, self.stop_x, self.start_y)
        self.set_axis_plots(self.ax_jersey, self.stop_x, self.start_y)

        # set up colors and patches for field
        self.set_up_field(endzones, linenumbers)
        
        # create scatterplots on axis for data
        self.scat_field = self.field_ax.scatter([], [], s=50, color='orange')
        self.scat_home = self.ax_home.scatter([], [], s=50, color='blue')
        self.scat_away = self.ax_away.scatter([], [], s=50, color='red')
        
        # add direction stats and jersey numbers/names
        self._scat_jersey_list = []
        self._scat_number_list = []
        self._scat_name_list = []
        self._a_dir_list = []
        self._a_or_list = []
        for _ in range(self.MAX_FIELD_PLAYERS):
            self._scat_jersey_list.append(self.ax_jersey.text(0, 0, '', horizontalalignment = 'center', verticalalignment = 'center', c = 'white'))
            self._scat_number_list.append(self.ax_jersey.text(0, 0, '', horizontalalignment = 'center', verticalalignment = 'center', c = 'black'))
            self._scat_name_list.append(self.ax_jersey.text(0, 0, '', horizontalalignment = 'center', verticalalignment = 'center', c = 'black'))
            
            self._a_dir_list.append(self.field_ax.add_patch(patches.Arrow(0, 0, 0, 0, color = 'k')))
            self._a_or_list.append(self.field_ax.add_patch(patches.Arrow(0, 0, 0, 0, color = 'k')))
        
        # return all axis plots that you want to update
        return (self.scat_field, self.scat_home, self.scat_away, *self._scat_jersey_list, *self._scat_number_list, *self._scat_name_list)
    
    def update(self, i):
        time_df = self.play_df.query("frameId == @i+1")
        
        #time_df['team_indicator'] = self.add_team_indicator(time_df)
        
        label_list = time_df.club.unique()
        label1= label_list[0]
        label2 = label_list[1]
        label3 = label_list[2]
        
        # update each team/football x,y coordinates
        for label in label_list:
            label_data = time_df[time_df.club == label]
            
            if label == label3:
                self.scat_field.set_offsets(label_data[['x','y']].to_numpy())
            elif label == label2:
                self.scat_home.set_offsets(label_data[['x','y']].to_numpy())
            elif label == label1:
                self.scat_away.set_offsets(label_data[['x','y']].to_numpy())
        
        #add direction and jersey info
        jersey_df = time_df[time_df.jerseyNumber.notnull()].reset_index()
        
        for (index, row) in jersey_df.iterrows():
            #self._scat_jersey_list[index].set_position((row.x, row.y))
            #self._scat_jersey_list[index].set_text(row.position)
            self._scat_number_list[index].set_position((row.x, row.y+1.9))
            self._scat_number_list[index].set_text(int(row.jerseyNumber))
            #self._scat_name_list[index].set_position((row.x, row.y-1.9))
            #self._scat_name_list[index].set_text(row.displayName.split()[-1])
            
            player_orientation_rad = self.deg_to_rad(self.convert_orientation(row.o))
            player_direction_rad = self.deg_to_rad(self.convert_orientation(row.dir))
            player_speed = row.s
            
            player_vel = np.array([np.real(self.polar_to_z(player_speed, player_direction_rad)), np.imag(self.polar_to_z(player_speed, player_direction_rad))])
            player_orient = np.array([np.real(self.polar_to_z(2, player_orientation_rad)), np.imag(self.polar_to_z(2, player_orientation_rad))])
            
            self._a_dir_list[index].remove()
            self._a_dir_list[index] = self.field_ax.add_patch(patches.Arrow(row.x, row.y, player_vel[0], player_vel[1], color = 'k'))
            
            self._a_or_list[index].remove()
            self._a_or_list[index] = self.field_ax.add_patch(patches.Arrow(row.x, row.y, player_orient[0], player_orient[1], color = 'grey', width = 2))
                

        return (self.scat_field, self.scat_home, self.scat_away, *self._scat_jersey_list, *self._scat_number_list, *self._scat_name_list)
    
    def set_up_field(self, endzones=True, linenumbers=True) -> None:
        yard_numbers_size = self.fig.get_size_inches()[0]*1.5

        # color field 
        rect = patches.Rectangle((0, 0), 120, 53.3, linewidth=0.1,
                                    edgecolor='r', facecolor='darkgreen', zorder=0)
        self.field_ax.add_patch(rect)

        # plot
        self.field_ax.plot([10, 10, 10, 20, 20, 30, 30, 40, 40, 50, 50, 60, 60, 70, 70, 80,
                    80, 90, 90, 100, 100, 110, 110, 120, 0, 0, 120, 120],
                    [0, 0, 53.3, 53.3, 0, 0, 53.3, 53.3, 0, 0, 53.3, 53.3, 0, 0, 53.3,
                    53.3, 0, 0, 53.3, 53.3, 0, 0, 53.3, 53.3, 53.3, 0, 0, 53.3],
                    color='white')

        # Endzones
        if endzones:
            ez1 = patches.Rectangle((0, 0), 10, 53.3,
                                    linewidth=0.1,
                                    edgecolor='r',
                                    facecolor='blue',
                                    alpha=0.2,
                                    zorder=0)
            ez2 = patches.Rectangle((110, 0), 120, 53.3,
                                    linewidth=0.1,
                                    edgecolor='r',
                                    facecolor='blue',
                                    alpha=0.2,
                                    zorder=0)
            self.field_ax.add_patch(ez1)
            self.field_ax.add_patch(ez2)
            
        if endzones:
            hash_range = range(11, 110)
        else:
            hash_range = range(1, 120)

        # add hashes
        for x in hash_range:
            self.field_ax.plot([x, x], [0.4, 0.7], color='white')
            self.field_ax.plot([x, x], [53.0, 52.5], color='white')
            self.field_ax.plot([x, x], [22.91, 23.57], color='white')
            self.field_ax.plot([x, x], [29.73, 30.39], color='white')
            
        # add linenumbers
        if linenumbers:
                for x in range(20, 110, 10):
                    numb = x
                    if x > 50:
                        numb = 120 - x
                    self.field_ax.text(x, 5, str(numb - 10),
                            horizontalalignment='center',
                            fontsize=yard_numbers_size,  # fontname='Arial',
                            color='white')
                    self.field_ax.text(x - 0.95, 53.3 - 5, str(numb - 10),
                            horizontalalignment='center',
                            fontsize=yard_numbers_size,  # fontname='Arial',
                            color='white', rotation=180)

        self.field_ax.set_xlim(self.start_x, self.stop_x)
        self.field_ax.set_ylim(0, self.start_y)
        self.field_ax.set_xticks(range(self.start_x,self.stop_x, 10))

    @staticmethod
    def set_axis_plots(ax, max_x, max_y) -> None:
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)

        ax.set_xlim([0, max_x])
        ax.set_ylim([0, max_y])
        
    @staticmethod
    def convert_orientation(x):
        return (-x + 90)%360
    
    @staticmethod
    def polar_to_z(r, theta):
        return r * np.exp( 1j * theta)
    
    @staticmethod
    def deg_to_rad(deg):
        return deg*np.pi/180
        
    # returns column of team indicator
    def add_team_indicator(self, play_df, game_id=None):
        
        if game_id == None:
            game_id = play_df.gameId.values[0]
        
        game_df = self.games_df.query("gameId == @game_id")
        home_team = game_df['homeTeamAbbr'].values[0]
        away_team = game_df['visitorTeamAbbr'].values[0]
        
        conditions = [
            (play_df.team == home_team),
            (play_df.team == away_team),
            (play_df.team == 'football')
        ]
        choices = [1,2,3]
        return np.select(conditions, choices)