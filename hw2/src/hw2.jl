module hw2

# y  1 2 3 4 5 6 7 8 9 10 x
# 1 |d|l|l|l|d|l|l|l|l|l|
# 2 |d|r|d|u|d|r|d|r|d|u|
# 3 |d|u|d|u|r|u|r|u|d|u|
# 4 |d|u|d|u|l|l|l|l|l|u|
# 5 |r|u|r|r|r|r|r|r|r|u|
# start: (4, 3)  (x, y)
# target: (2, 4)

# Calculate value function of each state for entire policy
# Iterate:
#     Follow eps-greedy for next action (top 2)
#     Update policy (if didn't take policy action)
#     Recompute value of current state based on next action
#       (note that this may decrease or increase based on eps-greedy and/or changing environment)
#     Take action

greet() = print("Hello World!")

end # module
