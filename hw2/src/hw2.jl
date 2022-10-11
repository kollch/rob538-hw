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
    best¹ = findall(x -> x == maximum(r), r)
    # If there are several, don't need to compare with eps
    if length(best¹) ≥ 2
        return rand(best¹)
    end
    best¹ = best¹[1]
    # Set the reward for the best to be as small as possible so we can find second-best
    r[best¹] = typemin(typeof(r[best¹]))
    # Find the second-best action
    best² = findall(x -> x == maximum(r), r)
    # If there's several second-best actions, choose one to work with
    best² = if length(best²) ≥ 2
        rand(best²)
    else
        best²[1]
    end
    # Are we choosing the second-best action?
    if rand() ≤ eps
        return best²
    end
    return best¹
end

# Run this function to see the scenario. `question` is either 1, 2, or 3, depending on the functionality required for the run.
function start_sim(question)
    reward = zeros(Int, 5, 10)
    reward[4, 2] = 20
    if question ≠ 1
        # Multiple agents and targets
        reward[1, 10] = 20
    end
    # Index of action for a position
    # Note that depending on the number of available actions, the number might be
    # different for the same action (e.g., 'down' is 1 when along the top row, but
    # 2 when in any other row)
    π₁ = [1 2 2 2 1 2 2 2 2 2
          2 4 2 1 2 4 2 4 2 1
          2 1 2 1 4 1 4 1 2 1
          2 1 2 1 3 3 3 3 3 1
          2 1 3 3 3 3 3 3 3 1]
    println("Original policy:")
    display(π₁)
    # Policy for the second robot
    π₂ = copy(π₁)
    π = [π₁, π₂]

    # Calculate value function of each state for entire policy
    # Value function for each state, based on the policy. Won't always be updated.
    pval₁ = if question == 1
        vals(1)
    else
        vals(2)
    end
    pval₂ = copy(pval₁)
    pval = [pval₁, pval₂]

    # Robot state: [Location, reward, done]
    robot₁ = [CartesianIndex(3, 4), 0, false]
    robot₂ = copy(robot₁)
    robots = if question == 1
        [robot₁]
    else
        [robot₁, robot₂]
    end
    global_reward = false
    # Iterate:
    # While not all robots are done:
    while !all(getindex.(robots, 3))
        for (i, robot) in enumerate(robots)
            # If done, don't do anything
            if robot[3]
                continue
            end
            println("Robot ", i, " current loc: ", Tuple(robot[1]))
            makemove!(robot, pval[i], π[i], reward, global_reward)
            global_reward = false
            if question == 3 && robot[3]
                # Just got done, so we must have hit a target
                global_reward = true
            end
        end
    end
    if question == 1
        println("Updated policy:")
        display(π[1])
    else
        println("Updated policies:")
        for πᵢ in π
            display(πᵢ)
        end
    end
end

function makemove!(robot, pval, π, reward, global_reward)
    # Follow ε-greedy for next action (top 2)
    actions = availactions(Tuple(robot[1])...)
    chosen_act = epsgreedy(0.1, robot[1], actions, pval)
    # Update policy (if didn't take policy action)
    π[robot[1]] = chosen_act
    # Recompute value of current state based on next action
    #   (note that this may decrease or increase based on ε-greedy and/or changing environment)
    pval[robot[1]] = pval[robot[1] + actions[chosen_act]] - 1
    if global_reward
        robot[2] += 20
        pval[robot[1]] += 20
    end
    # Take action
    robot[1] += actions[chosen_act]
    # Get any available rewards and take a timestep penalty
    robot[2] += reward[robot[1]] - 1
    println("New reward value: ", robot[2])
    if reward[robot[1]] == 20
        println("Found the target! Done.")
        # Hit a target, make it disappear
        reward[robot[1]] = 0
        # Done with this robot
        robot[3] = true
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
