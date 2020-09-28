use glam::Vec3;

#[derive(Debug)]
struct Stop {
    t: f32,
    rgb: Vec3,
}

impl Stop {
    pub fn new(t: f32, rgb: Vec3) -> Stop {
        Stop { t, rgb }
    }
}

#[derive(Debug)]
pub struct Gradient {
    stops: Vec<Stop>,
}

impl Gradient {
    pub fn linear_spacing(colors: &Vec<Vec3>) -> Gradient {
        // Must have at least one color to form a gradient
        assert!(colors.len() > 0);

        // Determine the step size between each stop
        let step_size = 1.0 / ((colors.len() - 1).max(1) as f32);
        let mut stops = vec![];

        for (index, rgb) in colors.iter().enumerate() {
            let stop = Stop::new(index as f32 * step_size, *rgb);
            println!("Stop added: {:?}", stop);

            stops.push(stop);
        }

        Gradient { stops }
    }

    pub fn empty() -> Gradient {
        Gradient { stops: Vec::new() }
    }

    pub fn add_stop(&mut self, t: f32, rgb: Vec3) {
        let stop = Stop::new(t, rgb);

        if self.stops.is_empty() {
            self.stops.push(stop);
        } else {
            if let Some(index) = self.stops.iter().position(|stop| stop.t > t) {
                self.stops.insert(index, stop);
            }
        }
    }

    pub fn color_at(&self, t: f32) -> Vec3 {
        if t >= 1.0 {
            return self
                .stops
                .last()
                .expect("Gradient must have at least one color")
                .rgb;
        } else if t <= 0.0 {
            return self
                .stops
                .first()
                .expect("Gradient must have at least one color")
                .rgb;
        }

        let stop_index = self.stops.iter().position(|stop| stop.t > t).unwrap();

        let start_index = stop_index - 1;
        let interpolant = (t - self.stops[start_index].t)
            / (self.stops[stop_index].t - self.stops[start_index].t);

        self.stops[start_index]
            .rgb
            .lerp(self.stops[stop_index].rgb, interpolant)
    }
}
