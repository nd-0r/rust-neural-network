use na::DVector;

pub trait DataItem {
    fn item_as_vector(&self) -> &DVector<f32>;
    fn label_as_vector(&self) -> &DVector<f32>;
}
