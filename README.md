# Connect-4-AI
A Connect 4 Agent that learns to play to a proficient level using Deep Learning - Exploring Deep Learning, Neural Networks, Reinforcement Learning, and Intelligence through looking at how we can use modern techniques to play board games in an intelligent (and non brute force) manner - My Submission for the St. Paul's High Master's Prize.

The project draws on papers by Mathew Lai and his Giraffe Deep Learning Chess Engine and Gerald Tesauro, author of TD-Gammon.
Due to the requisite that a submission of non-essay format be of comparable length to a 2500 word essay, descriptions will be kept succinct and links to sources provided.

Background:
In 1992 Tesauro created TD-Gammon, a computer backgammon program. TD-Gammon achieved a level of play just slightly below that of the top human backgammon players of the time, with Tesauro providing little to no prerequisite knowledge about the game. It explored strategies that humans had not pursued and made surprisingly effective moves, by using a Temporal Difference Algorithm sitting on top of, and updating the weights of, a neural network employing backwards propogating gradient descent.

Starting from random initial play, TD-Gammon's self-teaching methodology produced a surprisingly strong program: without lookahead, its positional judgement rivaled that of human experts, and when combined with shallow lookahead, it reached a level of play that surpassed even the best human players. Using this neural network to predict an evaluation function was really groundbreaking stuff and recent advancements such as the victory of Alpha Go over Lee Sedol using Deep Reinforcement Learning owe alot to Tesauro's work.

TD-Gammon was regarded as a breakthrough achievement in machine learning and computer game-playing research, and inspired numerous subsequent applications of reinforcement learning.

All the terms listed are extremely well documented.

https://medium.com/jim-fleming/before-alphago-there-was-td-gammon-13deff866197
https://web.stanford.edu/group/pdplab/pdphandbook/handbookch10.html
http://neuralnetworksanddeeplearning.com/chap2.html
https://deepmind.com/blog/deep-reinforcement-learning/

My Project:
My project looks to implement a similar design to that of Tesauro's to a game known as Connect-4. Again I use gradient descent and backpropogation in setting up my Artificial Neural Network (with two hidden layers so some board features can be realised), and TD-Learning is in fact applied, albeit in a more obscure manner and less mathematically rigorous way due to GPU limitations. The results of each game and the gamestate at each move of the game is stored in a connect-4.csv in a manner which apportions much greater weight to moves closer to the end of the game. The reward for loss is 0, while for winning this is 1. This is the subtle use of Reinforcement Learning. Predictions are made based on the stored memory of experiences of games, as well as using a part of a database of 8-ply games here to get it started (this will be deprecated shortly): https://archive.ics.uci.edu/ml/datasets/Connect-4

The memory() function handles storing the experience of the bot. By forcing the bot at any given step to predict using a regenerated model an evaluation score and choose the one with the best score, through repitition a better and better bot will evolve and has evolved. I have designed the program to be in some ways self inhibiting; it looks only one move ahead meaning there is absolutely at no level any brute forcing going on. In addition, the AI's response to immediate threats such as three-in-a-rows are not hardcoded in so initially while playing you will notice that the AI is not as adept at responding to attacking moves; however if you play alot more with it, my experience has been a marked understanding of the threat landscape and the need to make immediate advances. Training it against the random player will also help here. Ideas about centrality of board become more and more known to the AI as play continues although this may take some time, with the AI realising after significant training that creating towers in the more central columns early is advantageous in the end game against higher calibre opponents - some human play for sure motivates this.

Such is the excitement around AI today that I've made extensive use of modules like Tensorflow and Keras, so before running on your computer make sure either they are installed or email me and I will walk you through sshing into a Linux VM and spin one out on Microsoft Azure to take full advantage of multithreading and the immense number of available GPUs.

To play against the bot as a human go into the main function and uncomment the line that says: "humanInput = input("Enter column (0-6)")" and you will play as player 2 as "o". Note that on some computers running will be quite slow so be patient.

To train the model against a random legal move player, uncomment: humanInput = int(random.randint(0,6))

To train the model against a heuristic algorithm, uncomment the relevant lines.

Example output should look like:

[["b", "b", "b", "b", "b", "b", "b"],
 ["b", "b", "b", "b", "b", "b", "b"],
 ["b", "b", "b", "b", "b", "b", "b"],
 ["b", "b", "b", "x", "b", "b", "b"],
 ["b", "b", "o", "o", "b", "b", "b"],
 ["b", "x", "o", "x", "b", "b", "b"]]

Have Fun!

Each time a game ends a new one starts so do ctrl+c to move quit!
