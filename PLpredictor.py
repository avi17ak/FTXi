import pandas as pd                                     #library that displays data in Dataframes - easier to scrape and iterrate over rows/values
import matplotlib.pyplot as pyplot                      #allows data to be visualised
import csv                                              #library that allows reading,writing,appending etc. to csv files
import numpy as np                                      #library that is essential for data science as it crunches many numbers into a numpy array
from sklearn.utils import shuffle                       #class that allows me to decide what data to test and train
import sklearn                                          #machine learning library
from sklearn.model_selection import train_test_split    #class that splits data into train and testing sets
from tabulate import tabulate                           #Visualise data into a table
from sklearn import preprocessing                       #class that converts non-integer data types into appropriate integer numbers
import pickle                                           #Saves machine learning model into a .pickle file
from tkinter import *
from sklearn.neighbors import KNeighborsClassifier
import sqlite3



data = pd.read_csv('/Users/avkar/Documents/Allseasons.csv') #data for teams in the PL for the past 8 seasons into a dataframe
data = data[['HomeTeam','AwayTeam','FTHG','FTAG','FTR','HTHG','HTAG','HTR','HS','AS','HR','AR']]  #Attributes   FTHG = FULL TIME HOME GOALS
                                                                                                                #FTAG = FULL TIME AWAY GOALS
                                                                                                                #HTHG = HALF TIME HOME GOALS
                                                                                                                #HTAG = HALF TIME AWAY GOALS
                                                                                                                #HTR = HALF TIME RESULT
                                                                                                                #HS = HOME SHOTS
                                                                                                                #AS = AWAY SHOTS
                                                                                                                #HR = HOME REDS
                                                                                                                #AR = AWAY REDS

def team_ratings(data):
    Team_spends = {'Man United': [68.81,69.42 , 175.82, 140.40, 166.50, 178.56, 74.43], 'Bournemouth': [0, 0, 0, 49.60, 36.62, 30.87, 80.19],
                     'Fulham': [9.00, 26.55, 0, 0, 0, 0, 104.85], 'Huddersfield': [0, 0, 0, 0, 0, 51.08, 45.63],
                     'Newcastle': [28.77, 3.42, 40.52, 97.12, 0, 39.87, 53.78 ], 'Watford': [0, 0, 0, 75.18, 63.41, 64.26, 27.09],
                     'Wolves': [0, 0, 0, 0, 0, 0, 101.03],
                     'Arsenal': [50.40, 44.33, 107.08, 23.85,101.74, 137.57, 72.14 ], 'Liverpool': [63.54, 52.29, 136.29, 113.85, 71.91, 156.49, 163.98 ],
                     'Southampton': [37.35, 35.64, 86.40, 54.09, 62.01, 55.13, 56.03], 'Cardiff': [0, 41.23, 70, 70, 71, 70, 46.08],
                     'Chelsea': [98.73, 117.32, 123.93, 81.45, 119.52, 234.45, 187.92], 'Everton': [19.49, 28.62, 36.14, 44.01, 77.40, 182.88, 89.82],
                     'Leicester': [0, 0, 20.57, 44.91, 82.44, 78.93, 103.14], 'Tottenham': [65.93, 109.69, 43.63, 63.90, 75.15, 109.35, 0],
                     'West Ham': [21.51, 21.60, 31.64, 47.43, 75.15, 51.12, 84.51], 'Brighton': [0, 0, 0, 0, 0, 60.30, 79.52],
                     'Burnley': [0, 0, 11.36, 0, 41.04, 32.17, 29.70], 'Man City': [55.76, 104.40, 92.52, 187.47, 192.15, 285.75, 70.73],
                     'Crystal Palace': [0, 29.70, 28.10, 25.92, 91.17, 44.06, 10.67], 'Swansea': [18.42, 23.95, 37.38, 19.63, 52.29, 66.05, 0],
                     'Stoke': [21.31, 6.30, 1.62, 48.29, 34.90, 51.93, 0], 'West Brom': [4.50, 14.13, 22.39, 38.61, 34.11, 48.15, 0],
                     'Hull': [0, 27.99, 43.40, 0, 36.00, 0, 0], 'Middlesbrough': [0, 68, 0, 0, 43.16, 0, 0],
                     'Sunderland': [34.47, 30.52, 20.26, 59.45, 37.71, 0, 0], 'Wigan': [9.48, 0, 0, 0, 0, 0, 0],
                     'QPR': [45.32, 3.42, 39.18, 0, 0, 0, 0], 'Reading': [9.74, 0, 0, 0, 0, 0, 0],
                     'Aston Villa': [25.05, 17.86, 12.11, 59.90, 0, 0, 0], 'Norwich': [9.97, 27.05, 0, 42.80, 0, 0, 0], #Team spends in millions when in the premier league
                }                                                                                                       #In past seasons

    Team_overalls = {'Man United': [84, 82, 80, 79, 85, 83, 83, 81], 'Bournemouth': [64, 65, 67, 73, 73, 75, 75, 77], #Team ratings each past season
                     'Fulham': [77, 77, 69, 69, 71, 73, 76, 74], 'Huddersfield': [66, 66, 66, 67, 70, 73, 75, 72],    #measures how good the team is overall
                     'Newcastle': [78, 77, 75, 76, 74, 75, 76, 77], 'Watford': [68, 69, 69, 73, 76, 76, 77, 77],
                     'Wolves': [70, 67, 68, 69, 69, 73, 77, 78],
                     'Arsenal': [80, 79, 80, 80, 82, 83, 83, 81], 'Liverpool': [79, 79, 80, 80, 81, 82, 83, 84],
                     'Southampton': [72, 74, 75, 76, 77, 78, 77, 76], 'Cardiff': [69, 71, 70, 70, 71, 70, 73, 70],
                     'Chelsea': [81, 82, 82, 83, 83, 84, 84, 81], 'Everton': [76, 76, 79, 77, 80, 80, 79, 79],
                     'Leicester': [68, 67, 70, 74, 77, 78, 78, 78], 'Tottenham': [80, 80, 78, 77, 82, 83, 83, 83],
                     'West Ham': [73, 74, 74, 76, 78, 79, 78, 78], 'Brighton': [70, 69, 68, 70, 72, 74, 76, 76],
                     'Burnley': [65, 69, 69, 70, 73, 75, 77, 76], 'Man City': [82, 83, 82, 84, 84, 83, 85, 86],
                     'Crystal Palace': [69, 69, 72, 74, 78, 77, 77, 77], 'Swansea': [73, 75, 75, 76, 77, 76, 71, 70],
                     'Stoke': [75, 74, 75, 76, 78, 77, 72, 69], 'West Brom': [73, 74, 73, 74, 76, 77, 74, 71],
                     'Hull': [69, 72, 71, 72, 74, 70, 69, 68], 'Middlesbrough': [68, 68, 69, 71, 76, 72, 71, 69],
                     'Sunderland': [75, 74, 74, 75, 77, 71, 66, 66], 'Wigan': [72, 73, 69, 66, 68, 68, 69, 68],
                     'QPR': [76, 72, 75, 74, 72, 69, 69, 69], 'Reading': [71, 71, 68, 69, 70, 71, 69, 69],
                     'Aston Villa': [74, 72, 74, 75, 74, 73, 73, 75], 'Norwich': [72, 74, 72, 74, 74, 72, 71, 74],
                     }
    df = pd.DataFrame(data)

    HomeOvr = []
    AwayOvr = []
    HomeSpend = []
    AwaySpend = []
    df['HomeOvr'] = 0
    df['AwayOvr'] = 0
    df['HomeSpend'] = 0
    df['AwaySpend'] = 0
    for i, row in df.iterrows():                  #iterrates through rows of dataframe
        if i < 381:                               #index specific to the start of each season
            row['HomeOvr'] = Team_overalls[row['HomeTeam']][0]
            HomeOvr.append(row['HomeOvr'])  #appends the hometeam's and awayteam's fifa rating to that specific season
            row['AwayOvr'] = Team_overalls[row['AwayTeam']][0]
            AwayOvr.append(row['AwayOvr'])
            row['HomeSpend'] = Team_spends[row['HomeTeam']][0]
            HomeSpend.append(row['HomeSpend'])
            row['AwaySpend'] = Team_spends[row['AwayTeam']][0]
            AwaySpend.append(row['AwaySpend'])


        if i > 380 and i < 761:
            row['HomeOvr'] = Team_overalls[row['HomeTeam']][1]
            HomeOvr.append(row['HomeOvr'])
            row['AwayOvr'] = Team_overalls[row['AwayTeam']][1]
            AwayOvr.append(row['AwayOvr'])
            row['HomeSpend'] = Team_spends[row['HomeTeam']][1]
            HomeSpend.append(row['HomeSpend'])
            row['AwaySpend'] = Team_spends[row['AwayTeam']][1]
            AwaySpend.append(row['AwaySpend'])


        if i > 760 and i < 1141:
            row['HomeOvr'] = Team_overalls[row['HomeTeam']][2]
            HomeOvr.append(row['HomeOvr'])
            row['AwayOvr'] = Team_overalls[row['AwayTeam']][2]
            AwayOvr.append(row['AwayOvr'])
            row['HomeSpend'] = Team_spends[row['HomeTeam']][2]
            HomeSpend.append(row['HomeSpend'])
            row['AwaySpend'] = Team_spends[row['AwayTeam']][2]
            AwaySpend.append(row['AwaySpend'])

        if i > 1140 and i < 1521:
            row['HomeOvr'] = Team_overalls[row['HomeTeam']][3]
            HomeOvr.append(row['HomeOvr'])
            row['AwayOvr'] = Team_overalls[row['AwayTeam']][3]
            AwayOvr.append(row['AwayOvr'])
            row['HomeSpend'] = Team_spends[row['HomeTeam']][3]
            HomeSpend.append(row['HomeSpend'])
            row['AwaySpend'] = Team_spends[row['AwayTeam']][3]
            AwaySpend.append(row['AwaySpend'])
            #print('15/16', i, row['HomeSpend'], row['AwaySpend'])

        if i > 1520 and i < 1901:
            row['HomeOvr'] = Team_overalls[row['HomeTeam']][4]
            HomeOvr.append(row['HomeOvr'])
            row['AwayOvr'] = Team_overalls[row['AwayTeam']][4]
            AwayOvr.append(row['AwayOvr'])
            row['HomeSpend'] = Team_spends[row['HomeTeam']][4]
            HomeSpend.append(row['HomeSpend'])
            row['AwaySpend'] = Team_spends[row['AwayTeam']][4]
            AwaySpend.append(row['AwaySpend'])
            # print('16/17', i, row['HomeOvr'], row['AwayOvr'])

        if i > 1900 and i < 2281:
            row['HomeOvr'] = Team_overalls[row['HomeTeam']][5]
            HomeOvr.append(row['HomeOvr'])
            row['AwayOvr'] = Team_overalls[row['AwayTeam']][5]
            AwayOvr.append(row['AwayOvr'])
            row['HomeSpend'] = Team_spends[row['HomeTeam']][5]
            HomeSpend.append(row['HomeSpend'])
            row['AwaySpend'] = Team_spends[row['AwayTeam']][5]
            AwaySpend.append(row['AwaySpend'])
            # print('17/18', i, row['HomeOvr'], row['AwayOvr'])

        if i > 2280 and i < 2661:
            row['HomeOvr'] = Team_overalls[row['HomeTeam']][6]
            HomeOvr.append(row['HomeOvr'])
            row['AwayOvr'] = Team_overalls[row['AwayTeam']][6]
            AwayOvr.append(row['AwayOvr'])
            row['HomeSpend'] = Team_spends[row['HomeTeam']][6]
            HomeSpend.append(row['HomeSpend'])
            row['AwaySpend'] = Team_spends[row['AwayTeam']][6]
            AwaySpend.append(row['AwaySpend'])
            # print('18/19', i, row['HomeOvr'], row['AwayOvr'])



    df['HomeOvr'] = HomeOvr #Adds new columns to dataframes as a new attribute
    df['AwayOvr'] = AwayOvr
    df['HomeSpend'] = HomeSpend
    df['AwaySpend'] = AwaySpend

    return df






def explore(data):#Exploring Data:

    #win,loss,draw rate for home team



    h_wins = 0
    h_draws = 0
    h_loss = 0
    for i,row in data.iterrows():
        if row['FTR'] =='H':
            h_wins += 1         #adds a win to the home team if the result is a home win
        if row['FTR'] =='D':
            h_draws += 1        #adds a draw to home team and away tean if the result is a draw win
        if row['FTR'] == 'A':
            h_loss += 1

                #adds a win to the away team if the result is a away win
    size = [h_wins,h_draws,h_loss]
    labels = 'Home wins','Home draws','Home losses'
    colours = ['green','yellow','red']
    pyplot.pie(size,labels=labels,colors=colours,autopct='%1.1f%%',shadow=True)
    pyplot.show()   #displays pie chart that visualises the percentage of home wins, draws and away  wins of all games in the past 8 seasons

    #Most goals and points by a team in the past seasons
    with open('/Users/avkar/Documents/Allseasons.csv') as data:
        data = csv.DictReader(data) #reads through each row as tuples in a dictionary
                                    #used this rather than a normal reader as i can reference columns/attributes by their names rather than an index - easier to understand


        Clubs = {'Man United':0,'Bournemouth':0,'Fulham':0,'Huddersfield':0,'Newcastle':0,'Watford':0,'Wolves':0,
                     'Arsenal':0,'Liverpool':0,'Southampton':0,'Cardiff':0,'Chelsea':0,'Everton':0,
                     'Leicester':0,'Tottenham':0,'West Ham':0,'Brighton':0,'Burnley':0,'Man City':0,'Crystal Palace':0,
                     'Swansea':0,'Stoke':0,'West Brom':0,'Hull':0,'Middlesbrough':0,'Sunderland':0,'Wigan':0,'QPR':0,       #Dictionary that contains all clubs who've played in the
                     'Reading':0,'Aston Villa':0,'Norwich':0 }                                                              #premier league in the past 8 seasons and accumulates their total
                                                                                                                            # goals
        TeamPoints = {'Man United': 0, 'Bournemouth': 0, 'Fulham': 0, 'Huddersfield': 0, 'Newcastle': 0, 'Watford': 0,
                      'Wolves': 0,
                      'Arsenal': 0, 'Liverpool': 0, 'Southampton': 0, 'Cardiff': 0, 'Chelsea': 0, 'Everton': 0,
                      'Leicester': 0, 'Tottenham': 0, 'West Ham': 0, 'Brighton': 0, 'Burnley': 0, 'Man City': 0,            #Dictionary accumulating total points of teams
                      'Crystal Palace': 0,
                      'Swansea': 0, 'Stoke': 0, 'West Brom': 0, 'Hull': 0, 'Middlesbrough': 0, 'Sunderland': 0,
                      'Wigan': 0, 'QPR': 0,
                      'Reading': 0, 'Aston Villa': 0, 'Norwich': 0 }

        for line in data:                                           #iterates through rows of file
            Clubs[line['HomeTeam']] += int(line['FTHG'])            #Adds no. of goals scored in that game to total goals for that team(home and away)
            Clubs[line['AwayTeam']] += int(line['FTAG'])
            if line['FTR'] == 'D':
                TeamPoints[line['HomeTeam']] += 1                   #Adds no. of points earned depending on the FTR of game ; Draw = 1 point, Win= 3 points, Loss = 0 points
                TeamPoints[line['AwayTeam']] += 1
            if line['FTR'] == 'H':
                TeamPoints[line['HomeTeam']] += 3
            if line['FTR'] == 'A':
                TeamPoints[line['AwayTeam']] += 3

        TeamPoints = sorted(TeamPoints.items(),key=lambda t:t[1],reverse=True ) #sorts dicitionary from biggest to smallest keys in terms of their value size(values being points and goals)
        TeamGoals = sorted(Clubs.items(),key=lambda t:t[1],reverse=True )

        

       # headers = ['Team','Points']                                             #Table headers
       # headers2 =['Team', 'Goals']
       # print(tabulate(TeamPoints,headers = headers))                           #Puts the team points and team goals in an ordered table (bigest to smallest value
       # print("--------------  --------")
       # print(tabulate(TeamGoals,headers=headers2))

        return Clubs  #so i can pass the Clubs dictionary to another function

def score(var,var2,data):
    FTAG_pred = away_goals(var,var2,data)
    FTHG_pred = home_goals(var,var2,data)
    home_team = var.get()
    away_team = var2.get()
    window = Tk()
    lab = Label(window, font='helvetica 40 bold', bg='#FF015B', text=home_team + ' ' + str(FTHG_pred) + ' - ' + str(FTAG_pred) + ' '+away_team)
    img = PhotoImage(master=window, file='/Users/avkar/PycharmProjects/tensorEnv/pl_logo.png')
    lab['compound'] = BOTTOM
    lab['image'] = img
    lab.grid()
    window.mainloop()

def away_goals(var,var2,data):
    Clubs = explore(data)
    df = team_ratings(data)
    conv = preprocessing.LabelEncoder()
    HomeTeam = conv.fit_transform(list(data['HomeTeam']))
    AwayTeam = conv.fit_transform(list(data['AwayTeam']))
    HomeOvr = list(data['HomeOvr'])
    AwayOvr = list(data['AwayOvr'])
    HomeSpend = list(data['HomeSpend'])
    AwaySpend = list(data['AwaySpend'])
    FTR = conv.fit_transform(list(data['FTR']))
    HTHG = list(data['HTHG'])
    HTAG = list(data['HTAG'])
    FTHG = list(data['FTHG'])
    FTAG = list(data['FTAG'])
    HS = list(data['HS'])
    AS = list(data['AS'])
    X = list(zip(HomeTeam, AwayTeam, HomeOvr, AwayOvr, HomeSpend, AwaySpend, HTHG, HTAG, HS, AS))
    y = list(FTAG)
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1428, shuffle=False,
                                                                                stratify=None)
    away_team = var2.get()
    home_team = var.get()

   
    pickle_in = open("KNNmodelPlAwayGoals.pickle", "rb")

    KNN = pickle.load(pickle_in)
    predicted = KNN.predict(x_test)
    Teams = sorted(Clubs.keys())
    for x in range(len(predicted)):
        if home_team == Teams[x_test[x][0]] and away_team == Teams[x_test[x][1]]:
            FTAG_pred = predicted[x]

    return FTAG_pred

def home_goals(var,var2,data):
    Clubs = explore(data)
    df = team_ratings(data)
    conv = preprocessing.LabelEncoder()
    HomeTeam = conv.fit_transform(list(data['HomeTeam']))
    AwayTeam = conv.fit_transform(list(data['AwayTeam']))
    HomeOvr = list(data['HomeOvr'])
    AwayOvr = list(data['AwayOvr'])
    HomeSpend = list(data['HomeSpend'])
    AwaySpend = list(data['AwaySpend'])
    FTR = conv.fit_transform(list(data['FTR']))
    HTHG = list(data['HTHG'])
    HTAG = list(data['HTAG'])
    FTHG = list(data['FTHG'])
    FTAG = list(data['FTAG'])
    HS = list(data['HS'])
    AS = list(data['AS'])
    X = list(zip(HomeTeam, AwayTeam, HomeOvr, AwayOvr, HomeSpend, AwaySpend, HTHG, HTAG, HS, AS))
    y = list(FTHG)
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1428, shuffle=False,
                                                                                stratify=None)

    home_team = var.get()
    away_team = var2.get()
    pickle_in = open("KNNmodelPlHomeGoals.pickle", "rb")

    KNN = pickle.load(pickle_in)
    predicted = KNN.predict(x_test)
    Teams = sorted(Clubs.keys())
    for x in range(len(predicted)):
        if home_team == Teams[x_test[x][0]] and away_team == Teams[x_test[x][1]]:
            FTHG_pred = predicted[x]

    return FTHG_pred

def final_standings(team,data):
    df = team_ratings(data)
    Clubs = explore(data)
    window = Tk()
    window.configure(bg='#FF015B')
    conv = preprocessing.LabelEncoder()
    HomeTeam = conv.fit_transform(list(data['HomeTeam']))
    AwayTeam = conv.fit_transform(list(data['AwayTeam']))
    HomeOvr = list(data['HomeOvr'])
    AwayOvr = list(data['AwayOvr'])
    HomeSpend = list(data['HomeSpend'])
    AwaySpend = list(data['AwaySpend'])
    FTR = conv.fit_transform(list(data['FTR']))
    HTHG = list(data['HTHG'])
    HTAG = list(data['HTAG'])
    FTHG = list(data['FTHG'])
    FTAG = list(data['FTAG'])
    HS = list(data['HS'])
    AS = list(data['AS'])
    X = list(zip(HomeTeam,AwayTeam,HomeOvr,AwayOvr,HomeSpend,AwaySpend,HTHG,HTAG,HS,AS))
    y = list(FTR)
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1428, shuffle = False, stratify = None)
    '''
    ***OBTAINING OPTIMAL MACHINE LEARNING MODEL***
    
    train_acc = []
    test_acc = []
    number_neighbours = range(1, 15)
    best = 0

    # for neighbour in number_neighbours:
    
    for i in range(200000):
        x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1428)

        KNN = KNeighborsClassifier(n_neighbors=9)
        KNN.fit(x_train, y_train)

        accuracy = KNN.score(x_test, y_test)
        # test_acc.append(accuracy)
        # train_accuracy = KNN.score(x_train,y_train)
        # train_acc.append(train_accuracy)
        print('Model: ', i, accuracy)
        if accuracy > best:
            best = accuracy
            with open("KNNmodelPl.pickle", "wb") as file:
                pickle.dump(KNN, file)
    print("Best model: ", best)
    # pyplot.plot(number_neighbours, train_acc, label='Accuracy of the Training Set')
    # pyplot.plot(number_neighbours, test_acc, label='Accuracy of the Test Set')
    # pyplot.ylabel('Accuracy')
    # pyplot.xlabel('Number of Neighbors')
    # pyplot.legend()
    # pyplot.show()
    '''
    pickle_in = open("KNNmodelPL.pickle", "rb")
    KNN = pickle.load(pickle_in)
    predicted = KNN.predict(x_test)
    Teams = sorted(Clubs.keys())
    TeamPoints = {i:0 for i in Teams}
    unusedTeams = ['Hull','Sunderland','Middlesbrough','West Brom','Stoke','Swansea',
    'Wigan','QPR','Norwich','Aston Villa','Reading','Sheffield Utd']
    results = ['Away win','Draw    ','Home win']
    for x in range(len(predicted)):
        Home = Teams[x_test[x][0]]
        Away = Teams[x_test[x][1]]
        FTR_pred = results[predicted[x]]
        print("Predicted result: " + FTR_pred + "  Match: " + Home +' vs '+Away)
        if FTR_pred == 'Home win':
            TeamPoints[Home] += 3
        if FTR_pred == 'Away win':
            TeamPoints[Away] += 3
        if FTR_pred == 'Draw':
            TeamPoints[Home] += 1
            TeamPoints[Away] += 1
    for i in unusedTeams:
        if i in TeamPoints: del TeamPoints[i]

    def team_results(data):
        new_menu = Tk()
        new_menu.configure(bg='#FF015B')
        club = team[0][0]
        match = 0
        for x in range(len(predicted)):
            Home = Teams[x_test[x][0]]
            Away = Teams[x_test[x][1]]
            FTR_pred = results[predicted[x]]
            if Home == club or Away == club:
                match += 1
                Label(new_menu, text="Matchday " + str(match) + " Predicted result:  " + FTR_pred + "   Match: " + Home + '  vs  '+ Away, bg='#FF015B', font='helvetica 11 bold').grid(row=x)



        new_menu.mainloop()



    TeamPoints = sorted(TeamPoints.items(), key=lambda t: t[1], reverse=True)
    headers = ['Teams', 'Points']
    img = PhotoImage(master=window, file='/Users/avkar/PycharmProjects/tensorEnv/pl_logo.png')
    lab = Label(window, text=tabulate(TeamPoints, headers=headers), justify=RIGHT, font='helvetica 15', bg='#FF015B')
    lab['compound'] = RIGHT
    lab['image'] = img
    Button(window, text="See your team's game results", font='helvetica 25 bold', fg='#FF015B', bg='#FF015B', command=lambda:team_results(data)).grid()

    lab.grid()
    window.mainloop()

    return Clubs,unusedTeams,predicted,Home,Away,FTR_pred

def gui_team(team,data):
    window = Tk()
    window.configure(background = '#FF015B')
    team = team[0][0]
    team_wins = 0
    team_draws = 0
    team_losses = 0
    team_points = 0
    goals_scored = 0
    goals_conceded = 0
    for i,row in data.iterrows():
        if row['HomeTeam'] == team:
            goals_scored += row['FTHG']
            goals_conceded += row['FTAG']
            if row['FTR'] == 'H':
                team_wins += 1
                team_points += 3
            if row['FTR'] == 'D':
                team_draws += 1
                team_points += 1
            if row['FTR'] == 'A':
                team_losses += 1
        if row['AwayTeam'] == team:
            goals_scored += row['FTAG']
            goals_conceded += row['FTHG']
            if row['FTR'] == 'A':
                team_wins += 1
                team_points += 3
            if row['FTR'] == 'D':
                team_draws += 1
                team_points += 1
            if row['FTR'] == 'H':
                team_losses += 1
    team_games = team_wins + team_losses + team_draws
    win_rate = '%.2f' % ((team_wins / team_games) * 100)
    points_per_game = '%.2f' % (team_points / team_games)

    Label(window, bg='#FF015B', text=team + "'s " + ' statistics over the past 7 seasons', font='helvetica 20 bold').grid()
    Label(window, bg='#FF015B', text='Total games played: ' +str(team_games)).grid()
    Label(window, bg='#FF015B', text='Total points: ' +str(team_points)).grid()
    Label(window, bg='#FF015B', text='Points per game: ' +str(points_per_game)).grid()
    Label(window, bg='#FF015B', text='Total wins: '+str(team_wins)).grid()
    Label(window, bg='#FF015B', text='Total draws: ' + str(team_draws)).grid()
    Label(window, bg='#FF015B', text='Total losses: ' + str(team_losses)).grid()
    Label(window, bg='#FF015B', text='Total goals scored: ' + str(goals_scored)).grid()
    Label(window, bg='#FF015B', text='Total goals conceded: ' + str(goals_conceded)).grid()
    Label(window, bg='#FF015B', text='Win Percentage: ' + str(win_rate) + '%').grid()

    




def gui_predict():
    Clubs = explore(data)
    unusedTeams = ['Hull', 'Sunderland', 'Middlesbrough', 'West Brom', 'Stoke', 'Swansea',
                   'Wigan', 'QPR', 'Norwich', 'Aston Villa', 'Reading', 'Sheffield Utd']
    for i in unusedTeams:
        if i in Clubs: del Clubs[i]        # Deletes teams that weren't in the last recent premier league season
    menu = Tk()
    menu.configure(background='#FF015A')
    var = StringVar(menu,"Select Home Team")
    var2 = StringVar(menu,'Select Away Team')
    home = OptionMenu(menu, var, *Clubs)
    home.configure(background='#FF015B', font='helvetica 20')
    home.grid()
    away = OptionMenu(menu, var2, *Clubs)
    away.configure(background='#FF015B', font='helvetica 20')
    away.grid()
    button = Button(menu, font='helvetica 40 bold', width=20, text='Predict', command=lambda: [home_goals(var, var2, data), away_goals(var, var2, data), score(var,var2,data)]) #Calls functions in order it has been listed above
    button.configure(relief='groove')
    button.grid()
    img = PhotoImage(master=menu, file='/Users/avkar/PycharmProjects/tensorEnv/pl_logo.png')
    lab = Label(menu, bg='#FF015B')
    lab['compound'] = CENTER
    lab['image'] = img
    lab.grid()
    menu.mainloop()


    
def gui_menu(team):

    menu = Tk()
    menu.configure(background='#FF015B')
    mb = Menubutton(menu, text='Choose an option:', font=('Helvetica bold', 15), bg='#FF015B', fg='black')
    mb.menu = Menu(mb)
    mb['menu'] = mb.menu
    mb.menu.add_command(label='Match Predictor', command=lambda: gui_predict())
    mb.menu.add_command(label='Final Table Prediction', command=lambda: final_standings(team, data))  # Displays drop down of options user can choose from thbe app
    mb.menu.add_command(label='My Team Stats', command=lambda: gui_team(team, data))
    lab = Label(menu, text='Welcome to FTXi', fg='black',  font=('Helvetica bold', 25), bg='#FF015B')
    img = PhotoImage(master=menu, file='/Users/avkar/PycharmProjects/tensorEnv/tkinter bg.png')
    lab['compound'] = BOTTOM
    lab['image'] = img
    lab.pack()
    mb.pack()
    menu.mainloop()

def register():
    Clubs = explore(data)
    register_menu = Tk()
    register_menu.geometry("500x500")
    Label(register_menu, text="Welcome to FTXi, please register", font=('Helvetica bold', 25)).pack()
    fn = StringVar(register_menu)
    fn.set("Enter Firstname")
    ln = StringVar(register_menu)
    ln.set("Enter Surname")
    fn_entry = Entry(register_menu, textvariable = fn)  # Allows user to enter their firstname,surname,username and favourite team
    fn_entry.pack()
    ln_entry = Entry(register_menu, textvariable= ln)
    ln_entry.pack()
    un = StringVar(register_menu)
    un.set('Enter Username')
    un_entry = Entry(register_menu, textvariable=un)
    un_entry.pack()
    team = StringVar(register_menu, "Select Favourite Team")
    OptionMenu(register_menu, team, *Clubs).pack()              #Displays all premier league football teams in a drop down menu
    Button(register_menu, text='Submit', command=lambda: validate_register()).pack()  #Calls the validate_register function to check if registration meets requirements

    def validate_register(*args):
        user_name = un.get() #Obtains entries given by the user when submitting details in register subroutine
        firstname = fn.get()
        lastname = ln.get()
        club = team.get()


        database = sqlite3.connect('logins.db') #Establishes conncetion so database is opened
        cursor = database.cursor()
        cursor.execute("SELECT username FROM logins ")
        usernames = {row[0] for row in cursor.fetchall()}       #Iterates through username field to verify if username is taken or not
        if user_name in usernames:
            Label(register_menu, text = "Username taken").pack()
        if firstname == 'Enter Firstname' or lastname == 'Enter Surname':
            Label(register_menu, text="Please enter Firstname and Surname").pack()

        else:
            cursor.execute("INSERT INTO logins(first_name,last_name,username,team) VALUES (?,?,?,?)", (firstname, lastname, user_name, club,))
            database.commit()
            database.close()                                      #If all details are correct, the information is inserted into the database
            login()                                                #Cooncetion is closed and user is taken to login part of program


def login():
    def validate_login(*args):
        database = sqlite3.connect('logins.db')
        cursor = database.cursor()
        user_name = un.get()
        cursor.execute("SELECT username FROM logins ")      #Connects to databse and checks if the user name entered by the user exists,
        usernames = {row[0] for row in cursor.fetchall()}
        if user_name in usernames:
            team = cursor.execute('''SELECT team FROM logins WHERE username= ?''', (user_name,))
            team = cursor.fetchall()
            gui_menu(team)
        else:
            Label(login_menu,text = "Username incorrect or you have't signed up").pack()    #If it does, the user is allowed access to the app, else this error message is displayed

    login_menu = Tk()
    Label(login_menu, text="Welcome to FTXi, please login", font=('helvetica bold', 25)).pack()
    un = StringVar(login_menu, "Enter Username")
    Entry(login_menu, textvariable=un).pack()
    Button(login_menu, text='Enter', height='2', width='30', command=lambda: validate_login()).pack()    #Calls validate_login finction to see if the login exists in database
    Label(login_menu, text="New user?").pack()
    Button(login_menu, text='Create an account', height="2", width="30", command=lambda: register()).pack()  #Calls register function when user wants to create an account
    login_menu.mainloop()

login()
