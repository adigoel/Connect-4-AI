#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 22 03:35:24 2017
@author: adityagoel
"""

import sys
import numpy as np
import pandas as pd
import os
import math
import random


dataset2 = pd.read_csv('connect-4.csv')

X2 = dataset2.iloc[:, 1:43].values
Y2 = dataset2.iloc[:, 43].values

def memory(final,csvX,goes,X2,table,floor):
    if final == True:
        alpha = 0.9
        t = 0
        csvY = []
        if (goes%2) == 1:
            reward = 1
            print("\n"+"x wins"+"\n")
            for x in reversed(csvX):    
                u = alpha**t
                csvY.append(u*reward)
                t +=1
        else:
            print("\n"+"o wins"+"\n")
            reward = 0.5
            for x in csvX:    
                u = (alpha*0.8)**t
                csvY.append(u*reward)
                t +=1     
        
        if reward == 0.5:
            csvY = list(reversed(csvY))
        
        for a,b in enumerate(reversed(csvY)):
            csvX[a].append(b)
            
        i = 0
        try:
            while True:
                if csvX[i][-1] == csvX[i][-2]:
                    del csvX[i][-1]
                i=i+1
        except IndexError:
            pass
            
        for z in csvX:
            if type(z[-2])==float:
                del z[-1]
            z = [0]+z
            z = ",".join(map(str,z))            
            fd = open('connect-4.csv','a')
            fd.write("\n"+ z)
            fd.close()
        
        dataset2 = pd.read_csv('connect-4.csv')

        X2 = dataset2.iloc[:, 1:43].values
        Y2 = dataset2.iloc[:, 43].values
        ai = "x"
        human = "o"
        
        table = [["b", "b", "b", "b", "b", "b", "b"],
                 ["b", "b", "b", "b", "b", "b", "b"],
                 ["b", "b", "b", "b", "b", "b", "b"],
                 ["b", "b", "b", "b", "b", "b", "b"],
                 ["b", "b", "b", "b", "b", "b", "b"],
                 ["b", "b", "b", "b", "b", "b", "b"]]

        
        """
        table[rowdown][column right]
        """
        floor = [5 for x in range(7)]
        goes = 0
        turn = "x"
        csvX = []
        main(goes,turn,floor,table,False,X2,Y2,csvX)    
  
            
    if final == False:
        l = np.ndarray.tolist(X2[-1])
        csvX.append(l)
        return "d"

"""
def predict(b, X2, classifier, sc):
    a=np.ndarray.tolist(X2[0])
    print("kkk")
    predictions = []
    for x in b:
        x = [x]
        x = np.array(x)4
        predictions.append(classifier.predict(sc.transform(x)))
    h = [0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,1.000,0.000,0.000,1.000,1.000,0.000,0.000,1.000,1.000,0.000,0.000,1.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,1.000,0.000,0.000,1.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000]
    new_prediction = classifier.predict(sc.transform(np.array([h])))
    print(new_prediction)
    if new_prediction > 0.5:
        print("white wins")
    else:
        print("white loses")
    
predict(np.ndarray.tolist(X2[-3]))"""
        
def trainer(ingame,X2,a,Y2,goes,csvX):
    import numpy as np
    import pandas as pd
    
    if ingame == True:

        # Encoding categorical data
        from sklearn.preprocessing import LabelEncoder, OneHotEncoder
        
        
        for x in range(42):
            labelencoder = LabelEncoder()
            X2[:, x] = labelencoder.fit_transform(X2[:,x])
   
        i=0
        while True:
            try:
                onehotencoder = OneHotEncoder(categorical_features = [i])
                X2 = onehotencoder.fit_transform(X2).toarray()
                i = i+3
            except IndexError:
                break
                      
        i=0
        while True:
            try:
                X2 = np.delete(X2, (i), axis = 1)
        
                i = i+2
            except IndexError:
                break

        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X2, Y2, test_size = (0.001), random_state = 0)
        
        from sklearn.preprocessing import StandardScaler
        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)

        
        import keras
        from keras.models import Sequential
        from keras.layers import Dense
        
        classifier = Sequential()
        classifier.add(Dense(units = 42, kernel_initializer = 'uniform', activation = 'relu', input_dim = 84))
        classifier.add(Dense(units = 42, kernel_initializer = 'uniform', activation = 'relu'))
        classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
        classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
        classifier.fit(X_train, y_train, batch_size = 50, epochs = 2)
        y_pred = classifier.predict(X_test)
        y_pred = (y_pred > 0.5)
        predictions = []
        
        for u in range(-1,-8,-1):
            c=np.ndarray.tolist(X2[u])
            """a = X2[0]"""

            new_prediction = classifier.predict(sc.transform(np.array([c])))
            r = new_prediction[0][0]
    
            new_prediction[0][0] = new_prediction[0][0]/(42-goes)
            
            new_prediction[0][0] = 0.5-new_prediction[0][0]
            new_prediction[0] = np.absolute(new_prediction[0])
            if r>0.5:
                new_prediction[0][0]=new_prediction[0][0]+0.5
            elif r<0.5:
                new_prediction[0][0]=0.5-new_prediction[0][0]
            
                
            #print(new_prediction)
            np.append(Y2, new_prediction)
            predictions.append(new_prediction[0][0])

        predictions2 = []
        for a,b in enumerate(predictions):
            if goes < 35:
                b = b/(35-goes)
                b = b-0.5
                b=abs(b)

            predictions2.append(b)
            
        predictions = predictions2
        
        return(predictions)
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_test, y_pred)
    
    else:
        
        """DEPRECATED"""
        
        dataset2 = pd.read_csv('connect-4.csv')

        X2 = dataset2.iloc[:, 1:43].values
        Y2 = dataset2.iloc[:, 43].values
        gameStateCurrent = dataset1.iloc[:, 1:43].values
        
        from sklearn.preprocessing import LabelEncoder, OneHotEncoder

        for x in range(42):
            labelencoder = LabelEncoder()
            X2[:, x] = labelencoder.fit_transform(X2[:,x])

        i=0
        while True:
            try:
                onehotencoder = OneHotEncoder(categorical_features = [i])
                X2 = onehotencoder.fit_transform(X2).toarray()
                i = i+3
            except IndexError:
                break
                 
        i=0
        while True:
            try:
                X2 = np.delete(X2, (i), axis = 1)
        
                i = i+2
            except IndexError:
                break
         
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X2, Y2, test_size = (0.1), random_state = 0)
        
        from sklearn.preprocessing import StandardScaler
        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)

        import keras
        from keras.models import Sequential
        from keras.layers import Dense
        
        classifier = Sequential()
        classifier.add(Dense(units = 42, kernel_initializer = 'uniform', activation = 'relu', input_dim = 84))        
        classifier.add(Dense(units = 42, kernel_initializer = 'uniform', activation = 'relu'))
        classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
        classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
        classifier.fit(X_train, y_train, batch_size = 1, epochs = 4)
        
        y_pred = classifier.predict(X_test)
        y_pred = (y_pred > 0.5)
        
        a=np.ndarray.tolist(X2[3])
        
        b = [0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,1.000,0.000,0.000,1.000,1.000,0.000,0.000,1.000,1.000,0.000,0.000,1.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,1.000,0.000,0.000,1.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000]
        
        new_prediction = classifier.predict(sc.transform(np.array([a])))
        #print(new_prediction)
        if new_prediction > 0.5:
            print("white wins")
        else:
            print("white loses")
                 
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_test, y_pred)

        """  if ingame == True:
        np.seterr(invalid='ignore')
        listToPredict=[]
        print("sd")
        for x in range(1,k):
            listToPredict.append(np.ndarray.tolist(X2[(-1*x)]))
        u = len(listToPredict[0])
        print(u)
        predict(listToPredict, X2, classifier, sc)"""
    
def finished(table,csvX,goes,X2,floor):
    
    memory(True,csvX,goes,X2,table,floor)
    
def legitimateMoves(table,floor,X2,Y2,goes,csvX):
    """legitimate moves for o/AI written to the move.csv/top of the connect-4.csv file"""
    scores = []
    legitimateColumns = []
    try:
        i = 0
        while True:
            if floor[i] != -1:
                legitimateColumns.append(i)
            i=i+1
            
    except IndexError:
        pass
    #print(legitimateColumns)
    for x in legitimateColumns:
        
        table[floor[x]][x] = "x"
        gameArray = []
        for z in range(7):
            for i in range(5,-1,-1):
                gameArray.append(table[i][z])
    
    
        gameArray = np.asarray(gameArray)
        gameArray = np.reshape(gameArray, (1,42))
        #print(gameArray)
        np.append(X2, gameArray)

       
        
        if checkWin(table,x,True,floor,"x") == "win":
            finished(table,csvX,goes,X2,floor)

        elif checkWin(table,x,True,floor,"o") == "win":
            table[floor[x]][x] = "b"
            return(x)
        else:
            #print("norm")
            #print(a[0][0])
            table[floor[x]][x] = "b"
        #scores.append(a[0][0])
        #print(gameArray)

    a = trainer(True,X2,-1,Y2,goes,csvX)

    for x in a:
        scores.append(x)
          
    #print(scores)
    scores2 = []
    for x in legitimateColumns:
        scores2.append(scores[x])
    scores = scores2
    best = max(scores)        
    best_index = scores.index(best)
    col = legitimateColumns[best_index]
    #print(col)
    return(col)
    #floor[humanInput]=floor[humanInput]-1    

def competitorHeuristic(table,floor):
    evaluations = []
    legitimateColumns = []
    try:
        i = 0
        while True:
            if floor[i] != -1:
                legitimateColumns.append(i)
            i=i+1
            
    except IndexError:
        pass
    
    for x in legitimateColumns:
        table[floor[x]][x] = "o"
        twos = 0
        threes = 0
        fours = 0
        
        for x in range(5,-1,-1):
                for u in range(7):
                    try:
                        if table[x][u] == "b" or table[x][u] == "x":
                            pass
     
                        if table[x][u] == "o":
                            if table[x-1][u] == "o":
                                twos = twos+1
                                if table[x-2][u] == "o":
                                    threes = threes+1
                                    if table[x-3][u] == "o":
                    
                                        fours = fours+1
                            
                            if table[x][u+1] == "o":
                                twos = twos+1
                                if table[x][u+2] == "o":
                                    threes = threes+1
                                    if table[x][u+3] == "o":
                                        fours = fours+1
                            
                            if table[x-1][u+1] == "o":
                                twos = twos+1
                                if table[x-2][u+2] == "o":
                                    threes = threes+1
                                    if table[x-3][u+3] == "o":
                                        fours = fours+1
                            
                            if table[x+1][u+1] == "o":
                                twos = twos+1
                                if table[x+2][u+2] == "o":
                                    threes = threes+1
                                    if table[x+3][u+3] == "o":
                                        fours = fours+1
                            
                    except IndexError:
                        pass
                    
        evaluation = (10*twos) + (100*threes)
        evaluations.append(evaluation)
        if fours > 0:
                return(x)
        else:
            table[floor[x]][x] = "b"
            
    best = max(evaluations)        
    best_index = evaluations.index(best)
    
    if floor[best_index] == -1:
        del evaluations[best_index]
        
    best = max(evaluations)        
    best_index = evaluations.index(best)
    
    if floor[best_index] == -1:
        del evaluations[best_index]
        
    best = max(evaluations)        
    best_index = evaluations.index(best)
    
    if floor[best_index] == -1:
        del evaluations[best_index]
    
    
    c = legitimateColumns[best_index]   
    return(c)

def checkWin(table,lit,final,floor,typ):
    if final == False:
        for x in range(5,-1,-1):
            for u in range(7):
                try:
                    if table[x][u] == "b":
                        pass
 
                    if table[x][u] == "o":
                        if table[x-1][u] == "o":
                            pass
                            if table[x-2][u] == "o":
                                pass
                                if table[x-3][u] == "o":
                                    if x > 2:
                                        print("FD")
                                        main(goes,turn,floor,table,True,X2,Y2,csvX)
                                        """table[x][u],table[x-1][u],table[x-2][u],table[x-3][u]=table[x][u].upper(),table[x-1][u].upper(),table[x-2][u].upper(),table[x-3][u].upper()
                                        for a in table:
                                            print(a)"""
                        
                        if table[x][u+1] == "o":
                            pass
                            if table[x][u+2] == "o":
                                pass
                                if table[x][u+3] == "o":
                                    if u<4:
                                        
                                        main(goes,turn,floor,table,True,X2,Y2,csvX)                                        
                                        """table[x][u],table[x][u+1],table[x][u+2],table[x][u+3]=table[x][u].upper(),table[x][u+1].upper(),table[x][u+2].upper(),table[x][u+3].upper()
                                        for a in table:
                                            print(a)"""
                        
                        if table[x-1][u+1] == "o":
                            pass
                            if table[x-2][u+2] == "o":
                                pass
                                if table[x-3][u+3] == "o":
                                    if x>2:
                                        if u<4:
                                            main(goes,turn,floor,table,True,X2,Y2,csvX)                                            
                                            """table[x][u],table[x-1][u+1],table[x-2][u+2],table[x-3][u+3]=table[x][u].upper(),table[x-1][u+1].upper(),table[x-2][u+2].upper(),table[x-3][u+3].upper()
                                            for a in table:
                                                print(a)"""
                        
                        if table[x+1][u+1] == "o":
                            pass
                            if table[x+2][u+2] == "o":
                                pass
                                if table[x+3][u+3] == "o":
                                    if x<3:
                                        if u<4:
                                            main(goes,turn,floor,table,True,X2,Y2,csvX)                                            
                                            """table[x][u],table[x+1][u+1],table[x+2][u+2],table[x+3][u+3]=table[x][u].upper(),table[x-1][u+1].upper(),table[x-2][u+2].upper(),table[x-3][u+3].upper()
                                            for a in table:
                                                print(a)"""
                           
                    if table[x][u] == "x":
                        if table[x-1][u] == "x":
                            pass
                            if table[x-2][u] == "x":
                                pass
                                if table[x-3][u] == "x":
                                    if x > 2:
                                        print("FD")
                                        main(goes,turn,floor,table,True,X2,Y2,csvX)                                        
                                        """table[x][u],table[x-1][u],table[x-2][u],table[x-3][u]=table[x][u].upper(),table[x-1][u].upper(),table[x-2][u].upper(),table[x-3][u].upper()
                                        for a in table:
                                            print(a)"""
                        
                        if table[x][u+1] == "x":
                            pass
                            if table[x][u+2] == "x":
                                pass
                                if table[x][u+3] == "x":
                                    if u<4:
                                        print("FD")
                                        main(goes,turn,floor,table,True,X2,Y2,csvX)                                        
                                        """table[x][u],table[x][u+1],table[x][u+2],table[x][u+3]=table[x][u].upper(),table[x][u+1].upper(),table[x][u+2].upper(),table[x][u+3].upper()
                                        for a in table:
                                            print(a)"""
                        
                        if table[x-1][u+1] == "x":
                            pass
                            if table[x-2][u+2] == "x":
                                pass
                                if table[x-3][u+3] == "x":
                                    if x>2:
                                        if u<4:
                                            print("FD")
                                            main(goes,turn,floor,table,True,X2,Y2,csvX)                                            
                                            """table[x][u],table[x-1][u+1],table[x-2][u+2],table[x-3][u+3]=table[x][u].upper(),table[x-1][u+1].upper(),table[x-2][u+2].upper(),table[x-3][u+3].upper()
                                            for a in table:
                                                print(a)"""
                        
                        if table[x+1][u+1] == "x":
                            pass
                            if table[x+2][u+2] == "x":
                                pass
                                if table[x+3][u+3] == "x":
                                    if x<3:
                                        if u<4:
                                            print("FD")
                                            main(goes,turn,floor,table,True,X2,Y2,csvX)                                            
                                            """table[x][u],table[x+1][u+1],table[x+2][u+2],table[x+3][u+3]=table[x][u].upper(),table[x-1][u+1].upper(),table[x-2][u+2].upper(),table[x-3][u+3].upper()
                                            for a in table:
                                                print(a)"""
                        
                except IndexError:
                    pass
                    
    if final == True:
        #print("called")
        if lit == 999:
            pass
        else:
            table[floor[lit]][lit] = typ
        for x in range(5,-1,-1):
            for u in range(7):
                try:
                         
                    if table[x][u] == typ:
                        if table[x-1][u] == typ:
                            pass
                            if table[x-2][u] == typ:
                                pass
                                if table[x-3][u] == typ:
                                    if x > 2:
                                        table[floor[lit]][lit] = "b"
                                        return("win")
                                        """table[x][u],table[x-1][u],table[x-2][u],table[x-3][u]=table[x][u].upper(),table[x-1][u].upper(),table[x-2][u].upper(),table[x-3][u].upper()
                                        for a in table:
                                            print(a)"""
                        
                        if table[x][u+1] == typ:
                            pass
                            if table[x][u+2] == typ:
                                pass
                                if table[x][u+3] == typ:
                                    if u<4:
                                        table[floor[lit]][lit] = "b"
                                        return("win")
                                        
                                        """table[x][u],table[x][u+1],table[x][u+2],table[x][u+3]=table[x][u].upper(),table[x][u+1].upper(),table[x][u+2].upper(),table[x][u+3].upper()
                                        for a in table:
                                            print(a)"""
                        
                        if table[x-1][u+1] == typ:
                            pass
                            if table[x-2][u+2] == typ:
                                pass
                                if table[x-3][u+3] == typ:
                                    if x>2:
                                        if u<4:
                                            table[floor[lit]][lit] = "b"
                                            return("win")
                                            """table[x][u],table[x-1][u+1],table[x-2][u+2],table[x-3][u+3]=table[x][u].upper(),table[x-1][u+1].upper(),table[x-2][u+2].upper(),table[x-3][u+3].upper()
                                            for a in table:
                                                print(a)"""
                        
                        if table[x+1][u+1] == typ:
                            pass
                            if table[x+2][u+2] == typ:
                                pass
                                if table[x+3][u+3] == typ:
                                    if x<3:
                                        if u<4:
                                            table[floor[lit]][lit] = "b"
                                            return("win")
                                            """table[x][u],table[x+1][u+1],table[x+2][u+2],table[x+3][u+3]=table[x][u].upper(),table[x-1][u+1].upper(),table[x-2][u+2].upper(),table[x-3][u+3].upper()
                                            for a in table:
                                                print(a)"""
                        
                except IndexError:
                    pass 
        if lit in range(0,6):
            table[floor[lit]][lit] = "b"        
        return("nah")
    
    return("k")

  
ai = "x"
human = "o"

table = [["b", "b", "b", "b", "b", "b", "b"],
         ["b", "b", "b", "b", "b", "b", "b"],
         ["b", "b", "b", "b", "b", "b", "b"],
         ["b", "b", "b", "b", "b", "b", "b"],
         ["b", "b", "b", "b", "b", "b", "b"],
         ["b", "b", "b", "b", "b", "b", "b"]]

"""
table[rowdown][column right]
"""
floor = [5 for x in range(7)]

goes = 0
turn = "x"
csvX = []


def main(goes,turn,floor,table,won,X2,Y2,csvX):
        
    if won == True:
        memory(True,csvX,goes,X2,table,floor)
    
    if floor == [-1, -1, -1, -1, -1, -1]:
        for x in table:
                print(x)
        sys.exit("Draw!")
    
    if turn == "x":
        goes = goes + 1
        if goes == 1:
            humanInput = 3
            
        else:
            humanInput = legitimateMoves(table,floor,X2,Y2,goes,csvX)
 
        if humanInput != "narnia":
            humanInput = int(humanInput)
            if humanInput not in range(7) or humanInput == "":
                main(goes,turn,floor,table,False,X2,Y2,csvX)
            if floor[humanInput]==-1:
                print("Column full")
                #print(floor)
                main(goes,turn,floor,table,False,X2,Y2,csvX)
            if checkWin(table,99,False,floor,"o") == "k": 
                print("\n")
            
            table[floor[humanInput]][humanInput] = "x"
            floor[humanInput]=floor[humanInput]-1
            for x in table:
                print(x)
            
        gameArray = []
        for z in range(7):
            for i in range(5,-1,-1):
                gameArray.append(table[i][z])
        csvX.append(gameArray)
        
        my = True

    if turn == "o":
        goes = goes + 1
        
        """HERE IS WHERE TO CHANGE THE PLAYER METHOD AS SHOWN IN THE README"""
        humanInput = input("Enter column (0-6)")
        """humanInput = int(random.randint(0,6))"""
        """humanInput = competitorHeuristic(table,floor)"""
        print("Thinking")
        
        """
        if goes == 1:
            humanInput = 3
            
        else:
            humanInput = legitimateMoves(table,floor,X22,Y22,goes,csvX2)
        """    
        if humanInput != "narnia":
            humanInput = int(humanInput)
            if humanInput not in range(7):
                main(goes,turn,floor,table,False,X2,Y2,csvX)
            if floor[humanInput]==-1:
                print("Column full")
                #print(floor)
                main(goes,turn,floor,table,False,X2,Y2,csvX)
            if checkWin(table,99,False,floor,"x") == "k": 
                print("\n")
            
            table[floor[humanInput]][humanInput] = "o"
            floor[humanInput]=floor[humanInput]-1
            for x in table:
                print(x)
                  

        gameArray = []
        for z in range(7):
            for i in range(5,-1,-1):
                gameArray.append(table[i][z])
        csvX.append(gameArray)
        my = False
        
    if my == True:
        turn = "o"
    elif my == False:
        turn = "x"

    
    for x in range(5,-1,-1):
        for u in range(7):
            
            try:
                if table[x][u] == "x":
                    if table[x-1][u] == "x":
                        pass
                        if table[x-2][u] == "x":
                            pass
                            if table[x-3][u] == "x":
                                if x > 2:
                                    finished(table,csvX,goes,X2,floor)
                                    
                                    """table[x][u],table[x-1][u],table[x-2][u],table[x-3][u]=table[x][u].upper(),table[x-1][u].upper(),table[x-2][u].upper(),table[x-3][u].upper()
                                    for a in table:
                                        print(a)"""
                    
                    if table[x][u+1] == "o":
                        pass
                        if table[x][u+2] == "o":
                            pass
                            if table[x][u+3] == "o":
                                if u<4:
                                    finished(table,csvX,goes,X2,floor)
                                    
                                    """table[x][u],table[x][u+1],table[x][u+2],table[x][u+3]=table[x][u].upper(),table[x][u+1].upper(),table[x][u+2].upper(),table[x][u+3].upper()
                                    for a in table:
                                        print(a)"""
                    
                    if table[x-1][u+1] == "x":
                        pass
                        if table[x-2][u+2] == "x":
                            pass
                            if table[x-3][u+3] == "x":
                                if x>2:
                                    if u<4:
                                        finished(table,csvX,goes,X2,floor)
                                        
                                        """table[x][u],table[x-1][u+1],table[x-2][u+2],table[x-3][u+3]=table[x][u].upper(),table[x-1][u+1].upper(),table[x-2][u+2].upper(),table[x-3][u+3].upper()
                                        for a in table:
                                            print(a)"""
                    
                    if table[x+1][u+1] == "x":
                        pass
                        if table[x+2][u+2] == "x":
                            pass
                            if table[x+3][u+3] == "x":
                                if x<3:
                                    if u<4:
                                        finished(table,csvX,goes,X2,floor)
                                        
                                        """table[x][u],table[x+1][u+1],table[x+2][u+2],table[x+3][u+3]=table[x][u].upper(),table[x-1][u+1].upper(),table[x-2][u+2].upper(),table[x-3][u+3].upper()
                                        for a in table:
                                            print(a)"""
                
                                    
                        
                
                elif table[x][u] == "o":
                    if table[x-1][u] == "o":
                        pass
                        if table[x-2][u] == "o":
                            pass
                            if table[x-3][u] == "o":
                                if x > 2:
                                    finished(table,csvX,goes,X2,floor)
                                    
                                    """table[x][u],table[x-1][u],table[x-2][u],table[x-3][u]=table[x][u].upper(),table[x-1][u].upper(),table[x-2][u].upper(),table[x-3][u].upper()
                                    for a in table:
                                        print(a)"""
                    
                    if table[x][u+1] == "o":
                        pass
                        if table[x][u+2] == "o":
                            pass
                            if table[x][u+3] == "o":
                                if u<4:
                                    finished(table,csvX,goes,X2,floor)
                                    """table[x][u],table[x][u+1],table[x][u+2],table[x][u+3]=table[x][u].upper(),table[x][u+1].upper(),table[x][u+2].upper(),table[x][u+3].upper()
                                    for a in table:
                                        print(a)"""
                    
                    if table[x-1][u+1] == "o":
                        pass
                        if table[x-2][u+2] == "o":
                            pass
                            if table[x-3][u+3] == "o":
                                if x>2:
                                    if u<4:
                                        finished(table,csvX,goes,X2,floor)
                                        """table[x][u],table[x-1][u+1],table[x-2][u+2],table[x-3][u+3]=table[x][u].upper(),table[x-1][u+1].upper(),table[x-2][u+2].upper(),table[x-3][u+3].upper()
                                        for a in table:
                                            print(a)"""
                    
                    if table[x+1][u+1] == "o":
                        pass
                        if table[x+2][u+2] == "o":
                            pass
                            if table[x+3][u+3] == "o":
                                if x<3:
                                    if u<4:
                                        finished(table,csvX,goes,X2,floor)
                                        """table[x][u],table[x+1][u+1],table[x+2][u+2],table[x+3][u+3]=table[x][u].upper(),table[x-1][u+1].upper(),table[x-2][u+2].upper(),table[x-3][u+3].upper()
                                        for a in table:
                                            print(a)"""
                    
            except IndexError:
                pass
        
    main(goes,turn,floor,table,False,X2,Y2,csvX)

main(goes,turn,floor,table,False,X2,Y2,csvX)
