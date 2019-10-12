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
    def __init__(self, chatbot_name, username, init_emotional_state, init_emotional_history):
        # initialise variables
        self.username = username
        self.chatbot_name = chatbot_name
        # initialize all ui elements
        self.dgm = DiagramManager(init_emotional_state, init_emotional_history)
        self.root = tk.Tk()
        self.menubar = tk.Menu()
        self.file_menu = tk.Menu(tearoff=0)
        self.chatbot_menu = tk.Menu(tearoff=0)
        self.network_change_menu = tk.Menu(tearoff=0)
        self.load_menu = tk.Menu(tearoff=0)
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
        self.create_frame(chatbot_name)
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
        self.file_menu.add_cascade(label="Load character", menu=self.load_menu)
        self.load_menu.add_command(label="Load default character", command=lambda: self.forward_user_intent(intent="load_character", character="character_default"))
        self.load_menu.add_command(label="Load stable character", command=lambda: self.forward_user_intent(intent="load_character", character="character_stable"))
        self.load_menu.add_command(label="Load empathetic character", command=lambda: self.forward_user_intent(intent="load_character", character="character_empathetic"))
        self.load_menu.add_command(label="Load irascible character", command=lambda: self.forward_user_intent(intent="load_character", character="character_irascible"))
        self.file_menu.add_cascade(label="Load network", menu=self.network_change_menu)
        self.network_change_menu.add_command(label="FC, Emotion dataset, all features", command=lambda: self.forward_user_intent(intent="change_network", network="net_lin_emotion_all"))
        self.network_change_menu.add_command(label="FC, Tweet dataset, all features", command=lambda: self.forward_user_intent(intent="change_network", network="net_lin_tweet_all"))
        self.network_change_menu.add_command(label="LSTM+FC, Emotion dataset, all features", command=lambda: self.forward_user_intent(intent="change_network", network="net_rnn_emotion"))
        self.network_change_menu.add_command(label="LSTM+FC, Tweet dataset, all features", command=lambda: self.forward_user_intent(intent="change_network", network="net_rnn_tweet"))
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
    def forward_user_intent(self, intent, user_input=None, character=None, network=None):
        self.controller.handle_intent(intent=intent, input_message=user_input, character=character, network=network)

    # prints in chatout widget
    def update_chat_out(self, user_input, response):
        # prints input, empties input field
        self.chat_out.configure(state="normal")
        self.chat_out.insert(tk.END, self.username + ": " + user_input + "\n")
        # deletes text from index 0 till the end in input filed
        self.chat_in.delete(0, tk.END)
        # inserts chatbot answer in chat
        self.chat_out.insert(tk.END, self.chatbot_name + ": " + response + "\n")
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
        self.polar_angles = [n / float(5) * 2 * np.pi for n in range(5)]
        self.polar_angles += self.polar_angles[:1]
        self.polar_chart_yticks_positions = [0.2, 0.4, 0.6, 0.8]
        self.polar_chart_yticks_labels = [".2", ".4", ".6", ".8"]
        self.time_chart_x_values = [0, -1, -2, -3, -4]
        self.labels = []
        self.plot_colors = ["orange", "grey", "red", "blue", "green"]
        self.plot_colors_previous_step = ["black", "black", "black", "black", "black"]
        self.plot_classes = ["hap", "sad", "ang", "fea", "dis"]

        # 2D-lines that depict the development of the emotional state
        self.time_plot1, self.time_plot2, self.time_plot3, self.time_plot4, self.time_plot5 = (None, None, None, None, None)
        self.polar_plot = None
        self.bar_plot = None

        # init figure and subplots/axes
        self.fig = matplotlib.figure.Figure()
        self.ax3 = self.fig.add_subplot(211)
        self.ax4 = self.fig.add_subplot(212)
        # self.make_radar_chart(self.ax1, "Input emotions", 221, self.init_polar_data)
        # self.make_radar_chart(self.ax2, "Input keywords", 222, self.init_polar_data)

        # TODO hier damit ansetzen (siehe letztes todo)
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
        for index in range(len(self.plot_classes)):
            self.labels.append(self.plot_classes[index] + " (" + bar_data[index].__str__() + ")")

        ax.bar(self.labels, bar_data, width=.9, color=self.plot_colors, alpha=.75)
        ax.bar(self.labels, history_data[1], width=.01, color=self.plot_colors_previous_step, alpha=1)

    # create and update a line chart
    def make_time_chart(self, ax, init_time_data, title):
        ax.set_title(title)
        ax.yaxis.tick_right()
        ax.set_xlim(-4, 0)
        ax.set_xticks(np.arange(-4, 0, 1))
        ax.set_ylim(0, 1)
        ax.grid(axis="y", linestyle=':', linewidth=.5)
        # Graphen plotten
        self.time_plot1, = ax.plot(self.time_chart_x_values, [init_time_data[i][0] for i in range(0, 5)], linewidth=.5, color=self.plot_colors[0])
        self.time_plot2, = ax.plot(self.time_chart_x_values, [init_time_data[i][1] for i in range(0, 5)], linewidth=.5, color=self.plot_colors[1])
        self.time_plot3, = ax.plot(self.time_chart_x_values, [init_time_data[i][2] for i in range(0, 5)], linewidth=.5, color=self.plot_colors[2])
        self.time_plot4, = ax.plot(self.time_chart_x_values, [init_time_data[i][3] for i in range(0, 5)], linewidth=.5, color=self.plot_colors[3])
        self.time_plot5, = ax.plot(self.time_chart_x_values, [init_time_data[i][4] for i in range(0, 5)], linewidth=.5, color=self.plot_colors[4])
        # Legende erstellen
        ax.legend((self.time_plot1, self.time_plot2, self.time_plot3, self.time_plot4, self.time_plot5), self.plot_classes, loc=2)

    # create and update a radar chart
    def make_radar_chart(self, ax, title, position, polar_data):
        # Erstelle radar chart
        ax = plt.subplot(position, polar=True)
        ax.set_title(title)
        # Beschrifte die Achsen (x = der Kreis, y = die "Speichen"), ylim = limits der y-Achse
        plt.xticks(self.polar_angles[:-1], self.plot_classes, color='grey', size=10)
        plt.yticks(self.polar_chart_yticks_positions, self.polar_chart_yticks_labels, color="grey", size=8)
        ax.set_ylim(0, 1)
        # Plot data und fülle die Fläche dazwischen aus
        self.polar_plot, = ax.plot(self.polar_angles, polar_data, alpha=1, linewidth=5)
        ax.fill(self.polar_angles, polar_data, color='blue', alpha=0.1)

    def update_bar_chart(self, ax, emotional_state, history_data, canvas):
        ax.clear()
        self.make_bar_chart(ax, emotional_state, history_data, "emotional state")
        canvas.draw()

    def update_time_chart(self, time_data, diagram_canvas):
        self.time_plot1.set_ydata([time_data[i][0] for i in range(0, 5)])
        self.time_plot2.set_ydata([time_data[i][1] for i in range(0, 5)])
        self.time_plot3.set_ydata([time_data[i][2] for i in range(0, 5)])
        self.time_plot4.set_ydata([time_data[i][3] for i in range(0, 5)])
        self.time_plot5.set_ydata([time_data[i][4] for i in range(0, 5)])
        diagram_canvas.draw()

    def update_radar_chart(self, new_data, canvas):
        new_data[0].append(new_data[0][0])
        new_data[1].append(new_data[1][0])
        self.polar_plot.set_data(new_data[0], new_data[1])
        self.polar_plot.set_xdata(new_data[0])
        canvas.draw()

    def get_diagrams(self):
        return self.fig

    # old methods
    def old(self):
        # tutorial commands for using axes/suplots etc
        # Benutze diese Methode für polar-Diagramme
        # if method == 1:
        # Both, add_axes and add_subplot add an axes to a figure. They both return a matplotlib.axes.Axes object.

        # add_axes(x0, y0, width, height): Die Parameter geben die Position in der canvas ein
        # Beispiel unten zeichnet ein axes-Objekt von der linken unteren bis in die rechte obere ecke der canvas
        # Das axes-Objekt ist also genauso groß wie die canvas (parameter müssen eine liste sein)
        # Wenn man die Achsen/Labels sehen will, darf das axes nicht bei 0,0 staten. Labels/Achses sind nichT Teil
        # des axes-Objekts und sind daher nicht zu sehen wenn axes bei 0,0 startet
        #### self.ax =self.fig.add_axes([0, 0, 1, 1]).plot(self.x, self.y)

        # add,suplot(xxx): Man bestimmt wo das axes-objekt in einem virtuellen raster erscheint.
        # Bsp: add_suplot(123): Das Raster hat 1 Zeilen und 2 Spalten und man fügt das axes an der 3. Postion ein
        # Bsp: add_sbplot(111): Eine Zeile, eine Spalte. Axes nimmt also die ganze Flächer der canvas ein
        # Der Vorteil dieser Methode: plt macht die Positionierung und lässt genug Platz für die Labels der Achsen etc
        # Die sind nämlich nicht Teil des axes-Objektes und sind bei der Lösung oben daher nicht zu sehen
        # self.ax1 = self.fig.add_subplot(221)
        # self.ax2 = self.fig.add_subplot(222)
        # self.ax3 = self.fig.add_subplot(223)
        # self.ax4 = self.fig.add_subplot(224)

        # elif method == 2:
        # Den ganzen oberen Teil kann man auch einfacher haben:
        # Dabei werden die axes dem fig automatisch hinzugefügt
        #### self.fig, ((self.ax1, self.ax2), (self.ax3, self.ax4)) = plt.subplots(nrows=2, ncols=2)
        # nrows, ncols muss der Struktur der axes in den klammern ensprechen.
        # self.fig, ((self.ax1, self.ax2, self.ax3), (self.ax4, self.ax4, self.ax4,)) = plt.subplots(nrows=2, ncols=3)

        # In most cases, add_subplot would be the prefered method to create axes for plots on a canvas.
        # Only in cases where exact positioning matters, add_axes might be useful.
        pass
