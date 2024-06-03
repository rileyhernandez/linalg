

// fn main() {
//     let m = vec![vec![10., 15., 30., 10.],
//                  vec![5., 10., 10., 5.],
//                  vec![20., 20., 20., 20.],
//                  vec![35., 10., 100., 200.]];
//     let v = vec![vec![15.],
//                  vec![20.],
//                  vec![20.],
//                  vec![25.]];
//     // let mut mat = LinearSystem::new(m, v);
//     // println!("Starting matrix: ");
//     // mat.display();

//     let sol = LinearSystem::least_squares(m, v);
//     println!("Sol: ");
//     // sol.display();
//     println!("{:?}", sol);
// }

pub struct LinearSystem {
    matrix: Vec<Vec<f64>>
}

impl LinearSystem {
    pub fn new(matrix_a: Vec<Vec<f64>>, vector_b: Vec<Vec<f64>>) -> Self {
        let mut mat = matrix_a.clone();
        for row in 0..mat.len() {
            mat[row].push(vector_b[row][0]);
        };
        Self {matrix: mat}
    }

    pub fn display(&self) {
        let widths: Vec<usize> = (0..self.matrix[0].len())
            .map(|i| self.matrix.iter()
                .map(|row| format!("{:.2}", row[i]).len())
                .max()
                .unwrap_or(0))
            .collect();
    
        for row in &self.matrix {
            print!("| "); // Start of row
            for (i, &num) in row.iter().enumerate() {
                print!("{:width$.2} ", num, width = widths[i]);
            }
            println!("|"); 
        }
    }
    
    pub fn solve(&mut self) -> Vec<f64> {
        self.rref();
        self.matrix.reverse();
        let mut x = vec![self.matrix[0].last().unwrap().clone()];
        
        for row in 1..self.matrix.len() {
            let dot_product = LinearSystem::dot(
                &self.matrix[row][self.matrix.len()-row..],
                &x.iter().rev().cloned().collect::<Vec<_>>()
            );
            let value = self.matrix[row].last().unwrap() - dot_product;
            x.push(value);
        }
        self.matrix.reverse();
        x.reverse();
        x
    }
    

    fn rref(&mut self) {
        self.bring_up_nonzero_row();
        for row in 0..self.matrix.len() {
            self.normalize();
            self.subtract_rows(row);
        }
        self.normalize();
    }

    fn bring_up_nonzero_row(&mut self) {
        let nonzero_row = self.matrix.iter().position(|row| row[0] != 0.);
        self.matrix.rotate_left(nonzero_row.unwrap());
    }

    fn normalize(&mut self) {
        // iterate through rows of matrix, normalizing each to the leading nonzero element in each row
        for row in self.matrix.iter_mut() {
            if let Some(&first_nonzero) = row.iter().find(|&&x| x != 0.) {
                for elem in row.iter_mut() {
                    *elem = *elem / first_nonzero;
                }
            }
        }    
    }

    fn subtract_rows(&mut self, row_subtracted_from: usize) {
        if let Some(subtract_from) = self.matrix.get(row_subtracted_from).cloned() {
            for (index, row) in self.matrix.iter_mut().enumerate() {
                if index != row_subtracted_from {
                    for (col, elem) in row.iter_mut().enumerate() {
                        *elem -= subtract_from[col];
                    }
                }
            }
        }
    }

    fn dot(v1: &[f64], v2: &[f64]) -> f64 {
        v1.iter().zip(v2.iter()).map(|(a, b)| a * b).sum()
    }

    pub fn multiply(matrix_1: &Vec<Vec<f64>>, matrix_2: &Vec<Vec<f64>>) -> Vec<Vec<f64>> {
        if matrix_1[0].len() != matrix_2.len() {
            panic!("Matrix sizes incompatible!");
        }
        let mut new_matrix = vec![vec![0.; matrix_2[0].len()]; matrix_1.len()];
        for col in 0..matrix_2[0].len() {
            for row in 0..matrix_1.len() {
                new_matrix[row][col] = LinearSystem::dot(&matrix_1[row], &LinearSystem::transpose(&matrix_2)[col])
            }
        }
        new_matrix
    }

    pub fn transpose(matrix: &Vec<Vec<f64>>) -> Vec<Vec<f64>> {
        let mut new_matrix = vec![vec![0.; matrix.len()]; matrix[0].len()];
        // println!("DEBUG: {:?}", matrix);
        for row in 0..new_matrix.len()-1 {
            for col in 0..new_matrix[row].len()-1 {
                // println!("DEBUG: {}, {}", row, col);
                new_matrix[col][row] = matrix[row][col];
            }
        }
        new_matrix
    }

    pub fn least_squares(matrix_a: Vec<Vec<f64>>, vec_b: Vec<Vec<f64>>) -> Vec<f64> {
        let a_t_a = LinearSystem::multiply(&LinearSystem::transpose(&matrix_a), &matrix_a);
        let a_t_b = LinearSystem::multiply(&LinearSystem::transpose(&matrix_a), &vec_b);
        let mut system = LinearSystem::new(a_t_a, a_t_b);
        system.display();
        let solution = system.solve();
        solution
    }
}