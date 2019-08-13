import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import os

# Wenn ich hier ein NN aufbauen will, muss die Klasse von nn.Module erben
# die methoden forward und num_flat_features kommen von der vererbung und müssen implementiert werden
class Torch_Test(nn.Module):
    def __init__(self):
        super(Torch_Test, self).__init__()

        # Ein linearer Layer im NN
        # 10 = Eingabefeatures, 10 = Ausgabefeatures (features = neuronen)
        self.lin1 = nn.Linear(10, 10)
        # Diese Schicht muss die gleichen Anzahl INputfeatures haben, wie die letzt OUtput-Features
        self.lin2 = nn.Linear(10, 10)

    # nimmt den input x und rechnet ihn einmal komplett durch das netz
    # implementiert die forward propagation
    def forward(self, x):
        # man weist dem Layer eins die Aktivierungsfunktion relu zu
        # In x wird die Ausgabe von lin1 mit der Aktobierungsfunktion relu gespeichert
        x = F.relu(self.lin1(x))
        # Die letzte (Ausgabeschicht) hat keine Aktivuerzungfsfunktion
        x = self.lin2(x)
        return x



    def num_flat_features(self, x):
        size = x.size()[1:]
        num = 1
        for i in size:
            num *= i
        return num


netz = Torch_Test()
netz.cuda()
#print(netz)

#gucke ob ich schin ein gespeichertes netz habe
if os.path.isfile("dateiName.pt"):
    netz = torch.load("dateiName.pt")
    netz.cuda()
else:


# Variable baut aus einem Tensor eine variable
# DIe lin1/lin2 erwarten eine Eingabegröße von 10 (d.h. ein Vektor). Man erstellt aber eine Matrix mit 10*10
# Torch macht es dann automatisch so, dass es die Matrix zeilenweise nummt und alle Zeilen nacheinander ins Netz einspeist
for i in range(100):
    x = [1,0,0,1,0,1,0,1,1,0]
    input = Variable(torch.Tensor([x for _ in range(10)]))
    input = input.cuda()
    out = netz(input)
    #print(out)

    # Das ist der Zielvektor. Er enthält die gewünschten Werte für die Klassifikation (die label)
    # Ein WErt enstpricht hier einer Zeile in der input-Matrix
    # Das Netz soll das Label berechnet für eine EIngabe von 10 Features (WErten/NEuronen)
    target_list = [0,1,1,0,1,0,1,0,0,1]
    target = Variable(torch.Tensor([target_list for _ in range(10)]))
    target = target.cuda()
    # Wie soll der Fehler berechnet werden? (mean squared error)
    criterion = nn.MSELoss()
    # Hier wird der Fehler einer Eingabe und seines Targets berechnet
    loss = criterion(out, target)
    print(loss)
    # Gracienten auf 0 setzen, weil da noch werte aus dem vorigen trainingsschritt drin sein können
    netz.zero_grad()
    # loss durch backpropagation durchlaufen lassen
    loss.backward()
    # optimizer mit stochastic gradient descent
    # parameter aus dem netz, lr=learning rate
    optimizer = optim.SGD(netz.parameters(), lr=0.05)
    optimizer.step()

# Das Netz nach dem Durchlaufen speichern
torch.save(netz, "dateiName.pt")