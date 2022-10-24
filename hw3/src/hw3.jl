module hw3

export startsim

function takeaction!(agent)
	# Make chosen action the one with the highest estimated reward
	agent.action = argmax(agent.estreward)
end

dayreward(attended, b) = attended * exp(-attended / b)

systemreward(attend, b) = dayreward.(attend, b) |> sum

# Global reward
function getreward!(agent, sysreward)
	agent.reward = sysreward
end

# Simple local reward
function getreward!(agent, attended::Int, b)
	agent.reward = dayreward(attended, b)
end

# Difference reward
function getreward!(agent, sysstate, sysreward, b, c)
	syswithoutme = begin
		state = copy(sysstate)
		# Remove the agent from the system state
		state[agent.action] -= 1
		state
	end

	k = length(sysstate)
	cᵢ = if c == 1
		# cᵢ is nonexistance
		fill(0., k)
	else
		# cᵢ is splitting agent evenly across all days
		fill(1 / k, k)
	end
	# System without me + my default action cᵢ
	diffsys = syswithoutme .+ cᵢ

	# Difference reward is global - modified global
	agent.reward = sysreward - systemreward(diffsys, b)
end

function updateestreward!(agent)
	agent.estreward[agent.action] = agent.reward
end

function systemstate(agents, k)
	actions = map(a -> a.action, agents)
	[count(act -> act == day, actions) for day ∈ 1:k]
end

function startsim(weeks, n, b, k)
	# Use a global reward
	agents = [Agent(nothing, fill(1., k), 0.) for _ = 1:n]
	attendance₁ = fill(0, k)
	for week ∈ 1:weeks
		# Each agent chooses an action
		takeaction!.(agents)
		# Those actions lead to a system state
		sysstate = systemstate(agents, k)
		# The system state leads to a system reward
		sysreward = systemreward(sysstate, b)
		# Each agent receives a reward (i.e., agent reward)
		getreward!.(agents, sysreward) # Global reward
		updateestreward!.(agents)
		attendance₁ = sysstate
	end

	# Use a simple local reward
	agents = [Agent(nothing, fill(1., k), 0.) for _ = 1:n]
	attendance₂ = fill(0, k)
	for week ∈ 1:weeks
		# Each agent chooses an action
		takeaction!.(agents)
		# Those actions lead to a system state
		sysstate = systemstate(agents, k)
		# The system state leads to a system reward
		sysreward = systemreward(sysstate, b)
		# Each agent receives a reward (i.e., agent reward)
		getreward!.(agents, map(a -> sysstate[a.action], agents), b) # Simple local reward
		updateestreward!.(agents)
		attendance₂ = sysstate
	end

	# Use a difference reward with cᵢ being nonexistance
	agents = [Agent(nothing, fill(1., k), 0.) for _ = 1:n]
	attendance₃ = fill(0, k)
	for week ∈ 1:weeks
		# Each agent chooses an action
		takeaction!.(agents)
		# Those actions lead to a system state
		sysstate = systemstate(agents, k)
		# The system state leads to a system reward
		sysreward = systemreward(sysstate, b)
		# Each agent receives a reward (i.e., agent reward)
		getreward!.(agents, tuple(sysstate), sysreward, b, 1) # Difference reward
		updateestreward!.(agents)
		attendance₃ = sysstate
	end

	# Use a difference reward with cᵢ being a spread out action
	agents = [Agent(nothing, fill(1., k), 0.) for _ = 1:n]
	attendance₄ = fill(0, k)
	for week ∈ 1:weeks
		# Each agent chooses an action
		takeaction!.(agents)
		# Those actions lead to a system state
		sysstate = systemstate(agents, k)
		# The system state leads to a system reward
		sysreward = systemreward(sysstate, b)
		# Each agent receives a reward (i.e., agent reward)
		getreward!.(agents, tuple(sysstate), sysreward, b, 2) # Difference reward
		updateestreward!.(agents)
		attendance₄ = sysstate
	end

	println("Global reward: ", attendance₁)
	println("Simple local reward: ", attendance₂)
	println("Difference reward nonexistance: ", attendance₃)
	println("Difference reward spread out: ", attendance₄)
end

mutable struct Agent
	action::Union{Int,Nothing}
	estreward
	reward
end

end # module hw3
