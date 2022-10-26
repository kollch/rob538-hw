module hw3

using Gadfly
using DataFrames

export startsim

function takeaction!(agent)
	# Make chosen action ϵ-greedy (.2) with the highest estimated reward
	maxreward = argmax(agent.estreward)
	if rand() ≤ 0.01
		# All indices except for the one with max reward
		indices = [i for i ∈ 1:length(agent.estreward) if i ≠ maxreward]
		agent.action = rand(indices)
	else
		agent.action = maxreward
	end
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
	reward_df = DataFrame("Week" => Int[], "Reward Type" => String[], "System Reward" => Float64[])
	attend_df = DataFrame("Reward Type" => String[], "Day" => Int[], "Attended" => Int[])
	sum_df = DataFrame("Reward Type" => String[], "Day" => String[], "amin" => Int[], "amax" => Int[], "Average Attended" => Float64[])
	# Use a global reward
	agents = [Agent(nothing, fill(1., k), 0.) for _ = 1:n]
	attendance₁ = fill(0, k)
	amin = fill(1000, k)
	amax = fill(0, k)
	asum = fill(0, k)
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
		push!(reward_df, (week, "Global", sysreward))
		for (day, attended) ∈ enumerate(sysstate)
			amin[day] = min(amin[day], attended)
			amax[day] = max(amax[day], attended)
			asum[day] += attended
			push!(attend_df, ("Global", day, attended))
		end
		attendance₁ = sysstate
	end
	for day ∈ 1:k
		push!(sum_df, ("Global", "Day " * repr(day), amin[day], amax[day], asum[day] / weeks))
	end

	# Use a simple local reward
	agents = [Agent(nothing, fill(1., k), 0.) for _ = 1:n]
	attendance₂ = fill(0, k)
	amin = fill(1000, k)
	amax = fill(0, k)
	asum = fill(0, k)
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
		push!(reward_df, (week, "Local", sysreward))
		for (day, attended) ∈ enumerate(sysstate)
			amin[day] = min(amin[day], attended)
			amax[day] = max(amax[day], attended)
			asum[day] += attended
			push!(attend_df, ("Local", day, attended))
		end
		attendance₂ = sysstate
	end
	for day ∈ 1:k
		push!(sum_df, ("Local", "Day " * repr(day), amin[day], amax[day], asum[day] / weeks))
	end

	# Use a difference reward with cᵢ being `world without me`
	agents = [Agent(nothing, fill(1., k), 0.) for _ = 1:n]
	attendance₃ = fill(0, k)
	amin = fill(1000, k)
	amax = fill(0, k)
	asum = fill(0, k)
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
		push!(reward_df, (week, "Difference, without me", sysreward))
		for (day, attended) ∈ enumerate(sysstate)
			amin[day] = min(amin[day], attended)
			amax[day] = max(amax[day], attended)
			asum[day] += attended
			push!(attend_df, ("Difference, without me", day, attended))
		end
		attendance₃ = sysstate
	end
	for day ∈ 1:k
		push!(sum_df, ("Difference, without me", "Day " * repr(day), amin[day], amax[day], asum[day] / weeks))
	end

	# Use a difference reward with cᵢ being `world with average me`
	agents = [Agent(nothing, fill(1., k), 0.) for _ = 1:n]
	attendance₄ = fill(0, k)
	amin = fill(1000, k)
	amax = fill(0, k)
	asum = fill(0, k)
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
		push!(reward_df, (week, "Difference, with avg me", sysreward))
		for (day, attended) ∈ enumerate(sysstate)
			amin[day] = min(amin[day], attended)
			amax[day] = max(amax[day], attended)
			asum[day] += attended
			push!(attend_df, ("Difference, with avg me", day, attended))
		end
		attendance₄ = sysstate
	end
	for day ∈ 1:k
		push!(sum_df, ("Difference, with avg me", "Day " * repr(day), amin[day], amax[day], asum[day] / weeks))
	end

	println("Global reward:                  ", attendance₁, " (", systemreward(attendance₁, b), ")")
	println("Simple local reward:            ", attendance₂, " (", systemreward(attendance₂, b), ")")
	println("Difference reward, without me:  ", attendance₃, " (", systemreward(attendance₃, b), ")")
	println("Difference reward, with avg me: ", attendance₄, " (", systemreward(attendance₄, b), ")")

	#x = collect(1:length(attendance₁))
	#plot(x=x, y=attendance₁, Geom.bar, Guide.xlabel("xaxis"), Guide.ylabel("yaxis"))
	p₁ = plot(reward_df, x="Week", y="System Reward", color="Reward Type", Geom.line)
	p₂ = plot(reward_df, x="Reward Type", y="System Reward", Geom.violin)
	p₃ = plot(sum_df, x="Day", y="Average Attended", color="Reward Type", Scale.x_discrete(levels=["Day " * repr(i) for i ∈ 1:k]), Geom.bar(position=:dodge))
	display(p₁)
	display(p₂)
	display(p₃)
end

mutable struct Agent
	action::Union{Int,Nothing}
	estreward
	reward
end

end # module hw3
