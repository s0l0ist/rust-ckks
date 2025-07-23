use rand::distr::weighted::WeightedIndex;
use rand::prelude::*;

pub struct UniformRandomGenerator {
    rng: ThreadRng,
}

impl Default for UniformRandomGenerator {
    fn default() -> Self {
        Self::new()
    }
}

impl UniformRandomGenerator {
    pub fn new() -> Self {
        Self { rng: rand::rng() }
    }

    /// Sample choices by given weights. The greater their weights the more likely they get chosen.
    ///
    /// @param choices The to be selected samples
    /// @param weights Weights that get chosen by their weight/probability
    /// @return randomly selected sample by their weights
    pub fn weighted_choice(&mut self, choices: &[f64], weights: &[f64]) -> f64 {
        let dist = WeightedIndex::new(weights).unwrap();
        choices[dist.sample(&mut self.rng)]
    }
}
