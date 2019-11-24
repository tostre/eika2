import inspect
import tkinter as tk
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg as FigureCanvas

matplotlib.use('TkAgg')


class Frame:
    # initialize class instance
    def __init__(self, botname, init_emotional_state, init_emotional_history):
        # initialize all ui elements
        self.dgm = DiagramManager(init_emotional_state, init_emotional_history)
        self.root = tk.Tk()
        self.menubar = tk.Menu()
        self.file_menu = tk.Menu(tearoff=0)
        self.chatbot_menu = tk.Menu(tearoff=0)
        self.network_change_menu = tk.Menu(tearoff=0)
        self.load_character_menu = tk.Menu(tearoff=0)
        self.load_classifier_menu = tk.Menu(tearoff=0)
        self.load_logistic_regresssion_menu = tk.Menu(tearoff=0)
        self.load_random_forests_menu = tk.Menu(tearoff=0)
        self.load_neural_net_menu = tk.Menu(tearoff=0)
        self.load_classifier_emotion_menu  = tk.Menu(tearoff=0)
        self.load_classifier_tweet_menu = tk.Menu(tearoff=0)
        self.response_menu = tk.Menu(tearoff=0)
        self.chat_out = tk.Text(self.root, width=40, state="disabled")
        self.chat_in = tk.Entry(self.root)
        self.log = tk.Text(self.root, width=40, state="disabled")
        self.info_label = tk.Label(self.root, text="EIKA v.0.0.1, Marcel Müller, FH Dortmund ")
        self.send_button = tk.Button(self.root, text="Send", command=lambda: self.forward_user_intent(intent="get_response", user_input=self.chat_in.get()))
        self.diagram_frame = tk.Frame(self.root)
        self.diagram_canvas = FigureCanvas(self.dgm.get_diagrams(), master=self.diagram_frame)
        self.chat_in.bind(sequence='<Return>', func=lambda event: self.forward_user_intent(intent="get_response", user_input=self.chat_in.get()))
        # create frame and menu
        self.create_frame(botname)
        self.pack_widgets()
        # set subscriber list (implements observer pattern)
        self.controller = None
        self.subscribers = set()

    # creates main frame and menu bar
    def create_frame(self, title):
        # add menus to menubar
        self.menubar.add_cascade(label="File", menu=self.file_menu)
        self.menubar.add_cascade(label="Chatbot", menu=self.chatbot_menu)
        # create file menu
        self.file_menu.add_cascade(label="Load character", menu=self.load_character_menu)
        self.load_character_menu.add_command(label="Load default character", command=lambda: self.forward_user_intent(intent="load_character", character="character_default"))
        self.load_character_menu.add_command(label="Load stable character", command=lambda: self.forward_user_intent(intent="load_character", character="character_stable"))
        self.load_character_menu.add_command(label="Load empathetic character", command=lambda: self.forward_user_intent(intent="load_character", character="character_empathetic"))
        self.load_character_menu.add_command(label="Load irascible character", command=lambda: self.forward_user_intent(intent="load_character", character="character_irascible"))
        self.file_menu.add_cascade(label="Load classifier", menu=self.load_classifier_menu)
        # Load classifier menu
        self.load_classifier_menu.add_cascade(label="Logistic regression", menu=self.load_logistic_regresssion_menu)
        self.load_classifier_menu.add_cascade(label="Random Forests", menu=self.load_random_forests_menu)
        self.load_classifier_menu.add_cascade(label="Neural networks", menu=self.load_neural_net_menu)
        self.load_logistic_regresssion_menu.add_command(label="Emotion (Full)", command=lambda: self.forward_user_intent(intent="change_classifier", classifier_type="lr", dataset="norm_emotion", feature_set="full"))
        self.load_logistic_regresssion_menu.add_command(label="Emotion (Lex)", command=lambda: self.forward_user_intent(intent="change_classifier", classifier_type="lr", dataset="norm_emotion", feature_set="lex"))
        self.load_logistic_regresssion_menu.add_command(label="Tweet (Full)", command=lambda: self.forward_user_intent(intent="change_classifier", classifier_type="lr", dataset="norm_tweet", feature_set="full"))
        self.load_random_forests_menu.add_command(label="Emotion (Full)", command=lambda: self.forward_user_intent(intent="change_classifier", classifier_type="rf", dataset="norm_emotion", feature_set="full"))
        self.load_random_forests_menu.add_command(label="Emotion (Lex)", command=lambda: self.forward_user_intent(intent="change_classifier", classifier_type="rf", dataset="norm_emotion", feature_set="lex"))
        self.load_random_forests_menu.add_command(label="Tweet (Full)", command=lambda: self.forward_user_intent(intent="change_classifier", classifier_type="rf", dataset="norm_tweet", feature_set="full"))

        self.load_neural_net_menu.add_command(label="Emotion (Full)", command=lambda: self.forward_user_intent(intent="change_classifier", classifier_type="net", dataset="norm_emotion", feature_set="full"))
        self.load_neural_net_menu.add_command(label="Emotion (Lex)", command=lambda: self.forward_user_intent(intent="change_classifier", classifier_type="net", dataset="norm_emotion", feature_set="lex"))
        self.load_neural_net_menu.add_command(label="Tweet (Full)", command=lambda: self.forward_user_intent(intent="change_classifier", classifier_type="net", dataset="norm_tweet", feature_set="full"))
        self.load_neural_net_menu.add_command(label="Tweet (Lex)", command=lambda: self.forward_user_intent(intent="change_classifier", classifier_type="net", dataset="norm_tweet", feature_set="lex"))
        self.load_neural_net_menu.add_command(label="Tweet (Topics)", command=lambda: self.forward_user_intent(intent="change_classifier", classifier_type="net", dataset="norm_tweet", feature_set="topics"))

        # create debug menu
        self.chatbot_menu.add_command(label="Retrain chatbot", command=lambda: self.forward_user_intent(intent="retrain_bot"))
        self.chatbot_menu.add_command(label="Reset chatbot", command=lambda: self.forward_user_intent(intent="reset_state"))
        # configure frame
        self.root.configure(menu=self.menubar)
        self.root.title(title)
        self.root.resizable(0, 0)

    # places widgets in the frame
    def pack_widgets(self):
        # position ui elements
        self.chat_out.grid(row=1, column=1, columnspan=2, sticky=tk.E + tk.W + tk.N + tk.S)
        self.log.grid(row=1, column=3, sticky=tk.E + tk.W + tk.N + tk.S)
        self.chat_in.grid(row=2, column=1, sticky=tk.W + tk.E)
        self.send_button.grid(row=2, column=2, sticky=tk.E + tk.W)
        self.diagram_frame.grid(row=1, column=4, columnspan=2, sticky=tk.E + tk.W + tk.N + tk.S)
        self.info_label.grid(row=2, column=4, sticky=tk.E)
        self.diagram_canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

    # forwards user interaction with the gui to the controller
    def forward_user_intent(self, intent, user_input=None, character=None, classifier_type=None, dataset=None, feature_set=None):
        print(intent, classifier_type, dataset, feature_set)
        self.controller.handle_intent(intent, user_input, character, classifier_type, dataset, feature_set)

    # prints in chatout widget
    def update_chat_out(self, user_input, response, botname, username):
        # prints input, empties input field
        self.chat_out.configure(state="normal")
        self.chat_out.insert(tk.END, username + ": " + user_input + "\n")
        # deletes text from index 0 till the end in input filed
        self.chat_in.delete(0, tk.END)
        # inserts chatbot answer in chat
        self.chat_out.insert(tk.END, botname + ": " + response + "\n")
        self.chat_out.see(tk.END)
        self.chat_out.configure(state="disabled")

    # prints to the log widget, used to display additional text data (sentiment etc)
    def update_log(self, output_list, clear=True):
        self.log.configure(state="normal")
        if clear:
            self.log.delete(1.0, tk.END)

        # method expects dicts/lists/normal variables in a list and prints them
        for item in output_list:
            if isinstance(item, dict):
                for key, value in item.items():
                    self.log.insert(tk.END, key + ":\n")
                    self.log.insert(tk.END, value.__str__() + "\n\n")
            else:
                self.log.insert(tk.END, item.__str__() + "\n\n")

        self.log.config(state="disabled")

    # updates diagrams with new values
    def update_diagrams(self, emotional_state, emotional_history):
        self.dgm.update_time_chart(emotional_history, self.diagram_canvas)
        self.dgm.update_bar_chart(self.dgm.ax3, emotional_state, emotional_history, self.diagram_canvas)

    # lets other classes register themselves as observers
    def register_subscriber(self, who):
        # Set of subscribers, there should only ever be the controller in there
        self.subscribers.add(who)
        self.controller = who

    # draws the ui
    def show(self):
        self.root.mainloop()


class DiagramManager:
    def __init__(self, init_emotional_state, init_emotional_history):
        # Data that is needed to make the diagrams (labels, ticks, colors, etc)
        self.TIME_CHART_XTICKS = [0, -1, -2, -3, -4]
        self.PLOT_COLORS = ["orange", "grey", "red", "blue", "green"]
        self.PLOT_COLORS_PREVIOUS_STEPS = ["black", "black", "black", "black", "black"]
        self.PLOT_CLASSES = ["hap", "sad", "ang", "fea", "dis"]
        self.labels = []
        # 2D-lines that depict the development of the emotional state
        self.time_plot1, self.time_plot2, self.time_plot3, self.time_plot4, self.time_plot5 = (None, None, None, None, None)
        self.bar_plot = None
        # init figure and subplots/axes
        self.fig = matplotlib.figure.Figure()
        self.ax3 = self.fig.add_subplot(211)
        self.ax4 = self.fig.add_subplot(212)
        # create diagrams according to the visible diagrams
        self.make_bar_chart(self.ax3, init_emotional_state, init_emotional_history, "emotional state")
        self.make_time_chart(self.ax4, init_emotional_history, "emotional history")
        self.fig.set_tight_layout(True)

    # create and update a bar chart
    def make_bar_chart(self, ax, bar_data, history_data, title):
        ax.set_title(title)
        ax.set_ylim(0, 1)
        ax.yaxis.tick_right()
        ax.grid(axis="y", linestyle=':', linewidth=.5)

        self.labels.clear()
        for index in range(len(self.PLOT_CLASSES)):
            self.labels.append(self.PLOT_CLASSES[index] + " (" + bar_data[index].__str__() + ")")

        ax.bar(self.labels, bar_data, width=.9, color=self.PLOT_COLORS, alpha=.75)
        ax.bar(self.labels, history_data[1], width=.01, color=self.PLOT_COLORS_PREVIOUS_STEPS, alpha=1)

    # create and update a line chart
    def make_time_chart(self, ax, init_time_data, title):
        ax.set_title(title)
        ax.yaxis.tick_right()
        ax.set_xlim(-4, 0)
        ax.set_xticks(np.arange(-4, 0, 1))
        ax.set_ylim(0, 1)
        ax.grid(axis="y", linestyle=':', linewidth=.5)
        # Graphen plotten
        self.time_plot1, = ax.plot(self.TIME_CHART_XTICKS, [init_time_data[i][0] for i in range(0, 5)], linewidth=.5, color=self.PLOT_COLORS[0])
        self.time_plot2, = ax.plot(self.TIME_CHART_XTICKS, [init_time_data[i][1] for i in range(0, 5)], linewidth=.5, color=self.PLOT_COLORS[1])
        self.time_plot3, = ax.plot(self.TIME_CHART_XTICKS, [init_time_data[i][2] for i in range(0, 5)], linewidth=.5, color=self.PLOT_COLORS[2])
        self.time_plot4, = ax.plot(self.TIME_CHART_XTICKS, [init_time_data[i][3] for i in range(0, 5)], linewidth=.5, color=self.PLOT_COLORS[3])
        self.time_plot5, = ax.plot(self.TIME_CHART_XTICKS, [init_time_data[i][4] for i in range(0, 5)], linewidth=.5, color=self.PLOT_COLORS[4])
        # Legende erstellen
        ax.legend((self.time_plot1, self.time_plot2, self.time_plot3, self.time_plot4, self.time_plot5), self.PLOT_CLASSES, loc=2)

    # updates the data in the bar chart
    def update_bar_chart(self, ax, emotional_state, history_data, canvas):
        ax.clear()
        self.make_bar_chart(ax, emotional_state, history_data, "emotional state")
        canvas.draw()

    # update data in time chart
    def update_time_chart(self, time_data, diagram_canvas):
        self.time_plot1.set_ydata([time_data[i][0] for i in range(0, 5)])
        self.time_plot2.set_ydata([time_data[i][1] for i in range(0, 5)])
        self.time_plot3.set_ydata([time_data[i][2] for i in range(0, 5)])
        self.time_plot4.set_ydata([time_data[i][3] for i in range(0, 5)])
        self.time_plot5.set_ydata([time_data[i][4] for i in range(0, 5)])
        diagram_canvas.draw()

    # return diagrams
    def get_diagrams(self):
        return self.fig