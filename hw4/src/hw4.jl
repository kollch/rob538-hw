module hw4

export startsim

greet() = print("Hello World!")

function startsim()
	payoffs = [5 0
	           0 2]
	# Probability of Player 2 going to McMenamin's
	probₘ = 0.5
	# Estimated rewards
	Eᵣ = [0., 0.]
	for i ∈ 1:1000
		# Find which action the players went with
		p₁ = rand(1:2)
		p₂ = (rand() ≤ probₘ ? 2 : 1)
		reward = payoffs[p₁, p₂]
		# Update the estimated reward for that action
		Eᵣ[p₂] += (reward - Eᵣ[p₂]) / i
		# Adjust the probability based on the new estimated rewards
		probₘ += (argmax(Eᵣ) == 1 ? -0.001 : 0.001)
		# Make sure the probability isn't outside of the 0-1 range
		probₘ = clamp(probₘ, 0, 1)
		if probₘ == 0. || probₘ == 1.
			println("Finishing on iteration ", i)
			println("Final probability: ", probₘ)
			return
		end
	end
	println("Didn't converge.")
	println("Final probability: ", probₘ)
end

end # module hw4
