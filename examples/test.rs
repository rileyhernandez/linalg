use linalg::LinearSystem;
use linalg::Matrix;


fn main() {
    let m = vec![vec![10., 15., 30., 10.],
                 vec![5., 10., 10., 5.],
                 vec![20., 20., 20., 20.],
                 vec![35., 10., 100., 200.]];
    let v = vec![vec![15.],
                 vec![20.],
                 vec![20.],
                 vec![25.]];
    // let mut mat = LinearSystem::new(m, v);
    // println!("Starting matrix: ");
    // mat.display();

    let matrix = Matrix::Vecs(m);
    Matrix::display(&matrix);

    // let sol = LinearSystem::least_squares(m, v);
    // println!("Sol: ");
    // // sol.display();
    // println!("{:?}", sol);
}