module hw2

export start_sim

# Initial policy. 'u' = up, 'd' = down, 'l' = left, 'r' = right
# This policy has been chosen since it is a single cycle through all cells.
# i  1 2 3 4 5 6 7 8 9 10 j
# 1 |d|l|l|l|d|l|l|l|l|l|
# 2 |d|r|d|u|d|r|d|r|d|u|
# 3 |d|u|d|u|r|u|r|u|d|u|
# 4 |d|u|d|u|l|l|l|l|l|u|
# 5 |r|u|r|r|r|r|r|r|r|u|
# start (i, j): (3, 4)
# targets (i, j): (4, 2), (1, 10)

function vals(targets)
    if targets == 1
        # Target at (4, 2)
        [14  13  12  11  -8  -9 -10 -11 -12 -13
         15 -28 -27  10  -7  -4  -3   0   1 -14
         16 -29 -26   9  -6  -5  -2  -1   2 -15
         17  20 -25   8   7   6   5   4   3 -16
         18  19 -24 -23 -22 -21 -20 -19 -18 -17]
    else # 2 targets
        # Targets at (4, 2), (1, 10)
        [14 13 12 11 -8 -9 -10 -11 -12 20
         15  5  6 10 -7 -4  -3   0   1 19
         16  4  7  9 -6 -5  -2  -1   2 18
         17 20  8  8  7  6   5   4   3 17
         18 19  9 10 11 12  13  14  15 16]
    end
end

# Returns the index of the chosen action
function epsgreedy(eps, pos, actions, rewards)
    r = [rewards[pos + a] for a in actions]
    # Get all of the actions with the maximum reward
    best1 = findall(x -> x == maximum(r), r)
    # If there are several, don't need to compare with eps
    if length(best1) >= 2
        return rand(best1)
    end
    best1 = best1[1]
    # Set the reward for the best to be as small as possible so we can find second-best
    r[best1] = typemin(typeof(r[best1]))
    # Find the second-best action
    best2 = findall(x -> x == maximum(r), r)
    # If there's several second-best actions, choose one to work with
    best2 = if length(best2) >= 2
        rand(best2)
    else
        best2[1]
    end
    # Are we choosing the second-best action?
    if rand() > eps
        return best2
    end
    return best1
end

function start_sim()
    reward = zeros(Int, 5, 10)
    reward[4, 2] = 20
    # Index of action for a position
    # Note that depending on the number of available actions, the number might be
    # different for the same action (e.g., 'down' is 1 when along the top row, but
    # 2 when in any other row)
    policy1 = [1 2 2 2 1 2 2 2 2 2
               2 4 2 1 2 4 2 4 2 1
               2 1 2 1 4 1 4 1 2 1
               2 1 2 1 3 3 3 3 3 1
               2 1 3 3 3 3 3 3 3 1]
    # Policy for the second robot
    #policy2 = copy(policy1)

    # Calculate value function of each state for entire policy
    # Value function for each state, based on the policy. Won't always be updated.
    pval1 = vals(1)
    # Robot state: [Location, reward]
    robot1 = [CartesianIndex(3, 4), 0]
    # Iterate:
    while true
        println("Current loc: ", robot1[1])
        # Follow eps-greedy for next action (top 2)
        actions = availactions(Tuple(robot1[1])...)
        chosen_act = epsgreedy(0.9, robot1[1], actions, pval1)
        # Update policy (if didn't take policy action)
        policy1[robot1[1]] = chosen_act
        # Recompute value of current state based on next action
        #   (note that this may decrease or increase based on eps-greedy and/or changing environment)
        pval1[robot1[1]] = pval1[robot1[1] + actions[chosen_act]] - 1
        # Take action
        robot1[1] += actions[chosen_act]
        # Get any available rewards and take a timestep penalty
        robot1[2] += reward[robot1[1]] - 1
        println("New reward value: ", robot1[2])
        if reward[robot1[1]] == 20
            println("Found the target! Done.")
            # Hit a target, make it disappear
            reward[robot1[1]] = 0
            break
        end
    end
end

# Set up the actions list as a function, since they'll never change over time
function availactions(i, j)
    a = []
    # Up
    if i > 1
        push!(a, CartesianIndex(-1, 0))
    end
    # Down
    if i < 5
        push!(a, CartesianIndex(1, 0))
    end
    # Left
    if j > 1
        push!(a, CartesianIndex(0, -1))
    end
    # Right
    if j < 10
        push!(a, CartesianIndex(0, 1))
    end
    a
end

end # module
