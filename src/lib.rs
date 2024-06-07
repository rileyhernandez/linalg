use crate::MatrixError::EmptyVector;
use std::{error::Error, fmt};

pub struct LinearSystem {
    matrix: Vec<Vec<f64>>
}

pub enum Matrix {
    System(LinearSystem),
    Vecs(Vec<Vec<f64>>),
}

impl Matrix {
    pub fn display(system: &Matrix) {
        let matrix = match system {
            Matrix::System(linear_system) => &linear_system.matrix,
            Matrix::Vecs(vecs) => vecs,
        };

        let widths: Vec<usize> = (0..matrix[0].len())
            .map(|i| matrix.iter()
                .map(|row| format!("{:.2}", row[i]).len())
                .max()
                .unwrap_or(0))
            .collect();
    
        for row in matrix {
            print!("| "); // Start of row
            for (i, &num) in row.iter().enumerate() {
                print!("{:width$.2} ", num, width = widths[i]);
            }
            println!("|"); 
        }
    }
}

impl LinearSystem {
    pub fn new(matrix_a: Vec<Vec<f64>>, vector_b: Vec<Vec<f64>>) -> Result<Self, &'static str> {
        if matrix_a.len() != vector_b.len() {
            return Err("matrix_a and vector_b must have the same number of rows")
        }
        let rows_a = matrix_a.len();
        if !matrix_a.iter().all(|row| row.len() == rows_a) {
            return Err("matrix_a must be square")
        }

        let mut mat = matrix_a.clone();
        for row in 0..mat.len() {
            mat[row].push(vector_b[row][0]);
        }
        Ok(Self {matrix: mat})
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
    
    pub fn solve(&mut self) -> Vec<Vec<f64>> {
        // self.rref();
        // self.matrix.reverse();
        // let mut x = vec![self.matrix[0].last().unwrap().clone()];
        // 
        // for row in 1..self.matrix.len() {
        //     let dot_product = LinearSystem::dot(
        //         &self.matrix[row][self.matrix.len()-row..],
        //         &x.iter().rev().cloned().collect::<Vec<_>>()
        //     )?;
        //     let value = self.matrix[row].last().unwrap() - dot_product;
        //     x.push(value);
        // }
        // self.matrix.reverse();
        // x.reverse();
        // vec![x]
        vec![vec![69.]]
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

    // fn dot(v1: &[f64], v2: &[f64]) -> f64 {
    //     v1.iter().zip(v2.iter()).map(|(a, b)| a * b).sum()
    // }

    pub fn multiply(matrix_1: &Vec<Vec<f64>>, matrix_2: &Vec<Vec<f64>>) -> Result<Vec<Vec<f64>>, MatrixError> {
        let col_matrix_1 = match matrix_1.first() {
            Some(row_1) => row_1.len(),
            None => {
                return Err(MatrixError::InvalidMatrix)
            }
        };
        let row_matrix_2 = matrix_2.len();
        if col_matrix_1 != row_matrix_2 {
            return Err(MatrixError::InconsistentLengths)
        }

        let mut new_matrix = vec![vec![0.; matrix_2[0].len()]; matrix_1.len()];
        let matrix_2_transpose = LinearSystem::transpose(matrix_2);
        for col in 0..matrix_2_transpose.len() {
            for row in 0..matrix_1.len() {
                let matrix_1_row = vec![matrix_1[row].clone()];
                let matrix_2_col = vec![matrix_2_transpose[col].clone()];

                let matrix_1_row_t = LinearSystem::transpose(&matrix_1_row);
                let matrix_2_col_t = LinearSystem::transpose(&matrix_2_col);
                // println!("new mat: {:?}", new_matrix);
                // new_matrix[row][col] = matrix_1_row[0][0]*matrix_2_col[0][0];
                // new_matrix[row][col] = LinearSystem::dot(&matrix_1_row, &matrix_2_col)?;
                new_matrix[row][col] = LinearSystem::dot(&matrix_1_row_t, &matrix_2_col_t)?;
            }
        }
        Ok(new_matrix)
    }

    pub fn dot(vector_1: &Vec<Vec<f64>>, vector_2: &Vec<Vec<f64>>) -> Result<f64, MatrixError> {

        let vector_1_transpose = &LinearSystem::transpose(&vector_1.clone())[0];
        let vector_2_transpose = &LinearSystem::transpose(&vector_2.clone())[0];
        if vector_1_transpose.len() != vector_2_transpose.len() {
            return Err(MatrixError::InconsistentLengths);
        }
        let mut product = 0.;
        for elem in 0..vector_1_transpose.len() {
            product += vector_1_transpose[elem]*vector_2_transpose[elem];
        }
        Ok(product)
    }

    pub fn transpose(matrix: &Vec<Vec<f64>>) -> Vec<Vec<f64>> {
        let mut new_matrix = vec![vec![0.; matrix.len()]; matrix[0].len()];
        for row in 0..matrix.len() {
            for col in 0..matrix[row].len() {
                new_matrix[col][row] = matrix[row][col];
            }
        }
        new_matrix
    }

    pub fn unpack(matrix: Vec<Vec<f64>>) -> Result<f64, MatrixError> {
        match matrix.first() {
            Some(inner_vec) => match inner_vec.first() {
                Some(&value) if inner_vec.len() == 1 => Ok(value),
                Some(_) => Err(EmptyVector),
                None => Err(EmptyVector),
            },
            None => Err(EmptyVector),
        }
    }

    // pub fn least_squares(matrix_a: Vec<Vec<f64>>, vec_b: Vec<Vec<f64>>) -> Result<Vec<f64>, MatrixError> {
    //     let a_t_a = LinearSystem::multiply(
    //                                                                         &LinearSystem::transpose(&matrix_a),
    //                                                                         &matrix_a);
    //     let a_t_b = LinearSystem::multiply(
    //                                                                         &LinearSystem::transpose(&matrix_a), 
    //                                                                         &vec_b);
    //     let mut system = LinearSystem::new(a_t_a, a_t_b)?;
    //     Matrix::display(&system);
    //     let solution = system.solve();
    //     Ok(solution)
    // }

    // pub fn validate_matrix()
    }



#[derive(Debug)]
pub enum MatrixError {
    InconsistentLengths,
    InvalidMatrix,
    EmptyVector,
}

impl fmt::Display for MatrixError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Matrix operation failed")
    }
}
impl Error for MatrixError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        None
    }
}

// #[test]
// fn create_matrix() {
//     let m = vec![vec![10., 15., 30., 10.],
//                  vec![5., 10., 10., 5.],
//                  vec![20., 20., 20., 20.],
//                  vec![35., 10., 100., 200.]];
//     let _v = vec![vec![15.],
//                  vec![20.],
//                  vec![20.],
//                  vec![25.]];
//     let matrix = Matrix::Vecs(m);
//     Matrix::display(&matrix);
// }

#[test]
fn create_linear_system() {
    let m = vec![
                 vec![10., 15., 30., 10.],
                 vec![5., 10., 10., 5.],
                 vec![20., 20., 20., 20.],
                 vec![35., 10., 100., 200.]];
    let v = vec![
                 vec![15.],
                 vec![20.],
                 vec![20.],
                 vec![25.]];
    let system = LinearSystem::new(m, v).expect("Couldn't construct system");
    let intended = vec![
        vec![10., 15., 30., 10., 15.],
        vec![5., 10., 10., 5., 20.],
        vec![20., 20., 20., 20., 20.],
        vec![35., 10., 100., 200., 25.]];
}

// #[test]
// fn rref_system() {
//     let m = vec![vec![10., 15., 30., 10.],
//                  vec![5., 10., 10., 5.],
//                  vec![20., 20., 20., 20.],
//                  vec![35., 10., 100., 200.]];
//     let v = vec![vec![15.],
//                  vec![20.],
//                  vec![20.],
//                  vec![25.]];
//     let mut system = LinearSystem::new(m, v).expect("Couldn't construct system");
//     println!("Starting matrix: ");
//     system.display();
//     system.rref();
//     println!("RREF matrix: ");
//     system.display();
// }

#[test]
fn transpose_vector() {
    let a = vec![
        vec![1.],
        vec![2.],
        vec![3.],
    ];
    let a_t = LinearSystem::transpose(&a);
    assert_eq!(a_t, vec![vec![1., 2., 3.]]);
}

#[test]
fn multiply_vectors() {
    let a = vec![

        vec![1., 2., 3.]
    ];
    let b = vec![
        vec![1.],
        vec![1.],
        vec![1.],
    ];
    let product = LinearSystem::multiply(&a, &b).expect("failed mult");
    assert_eq!(product[0][0], 6.)
}

#[test]
fn dot_vectors() {
    let a = vec![
        vec![1.],
        vec![0.],
        vec![0.],
    ];
    let b = vec![
        vec![1.],
        vec![4.],
        vec![7.]
    ];
    let product = LinearSystem::dot(&a, &b).expect("Could not multiply");
    assert_eq!(product, 1.);
}

#[test]
fn identity_matrix() {
    let a = vec![
        vec![1., 0., 0.],
        vec![0., 1., 0.],
        vec![0., 0., 1.],
    ];
    let b = vec![
        vec![1., 2., 3.],
        vec![4., 5., 6.],
        vec![7., 8., 9.]
    ];
    let product = LinearSystem::multiply(&a, &b).expect("identity test");
    assert_eq!(product, b);
}

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
//
//     let matrix = Matrix::Vecs(m);
//     Matrix::display(&matrix);
//
//     // let sol = LinearSystem::least_squares(m, v);
//     // println!("Sol: ");
//     // // sol.display();
//     // println!("{:?}", sol);
// }