use glm::vec2;
use glm::vec3;
use mimalloc::MiMalloc;
use na::DMatrix;
pub use nalgebra as na;
pub use nalgebra_glm as glm;
use rand::Rng;
use rayon::prelude::*;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

#[macro_use]
extern crate bitflags;
use std::{
    alloc::dealloc,
    cell::{RefCell, UnsafeCell},
    collections::{HashMap, LinkedList},
    env,
    hash::Hasher,
    mem::MaybeUninit,
    ops::{Deref, Index, IndexMut, Mul},
    sync::{
        atomic::{AtomicPtr, AtomicU32, Ordering},
        Arc, RwLock,
    },
    usize,
};

use crate::nn::{mlp::Dense, Relu, SGD};

type Float = f64;
type Vec3 = glm::TVec3<Float>;
type Vec2 = glm::TVec2<Float>;
type Mat4 = glm::TMat4<Float>;
type Mat3 = glm::TMat3<Float>;

pub fn uvec2(x: u32, y: u32) -> glm::UVec2 {
    glm::UVec2::new(x, y)
}
pub fn uvec3(x: u32, y: u32, z: u32) -> glm::UVec3 {
    glm::UVec3::new(x, y, z)
}
pub struct AtomicFloat {
    bits: AtomicU32,
}
impl Default for AtomicFloat {
    fn default() -> Self {
        Self::new(0.0)
    }
}
impl AtomicFloat {
    pub fn new(v: f32) -> Self {
        Self {
            bits: AtomicU32::new(bytemuck::cast(v)),
        }
    }
    pub fn load(&self, ordering: Ordering) -> f32 {
        bytemuck::cast(self.bits.load(ordering))
    }
    pub fn store(&self, v: f32, ordering: Ordering) {
        self.bits.store(bytemuck::cast(v), ordering)
    }
    pub fn fetch_add(&self, v: f32, ordering: Ordering) -> f32 {
        let mut oldbits = self.bits.load(ordering);
        loop {
            let newbits: u32 = bytemuck::cast(bytemuck::cast::<u32, f32>(oldbits) + v);
            match self.bits.compare_exchange_weak(
                oldbits,
                newbits,
                Ordering::SeqCst,
                Ordering::SeqCst,
            ) {
                Ok(_) => break,
                Err(x) => oldbits = x,
            }
        }
        bytemuck::cast(oldbits)
    }
}
impl Clone for AtomicFloat {
    fn clone(&self) -> Self {
        Self {
            bits: AtomicU32::new(self.bits.load(Ordering::Relaxed)),
        }
    }
}
#[derive(Debug, Clone, Copy)]
pub struct Bound2<T: na::Scalar> {
    pub min: glm::TVec2<T>,
    pub max: glm::TVec2<T>,
}

#[derive(Debug, Clone, Copy)]
pub struct Bound3<T: na::Scalar> {
    pub min: glm::TVec3<T>,
    pub max: glm::TVec3<T>,
}

impl<T> Bound3<T>
where
    T: glm::Number,
{
    pub fn size(&self) -> glm::TVec3<T> {
        self.max - self.min
    }

    pub fn insert_point(&mut self, p: &glm::TVec3<T>) {
        self.min = glm::min2(&self.min, p);
        self.max = glm::max2(&self.max, p);
    }
    pub fn insert_box(&mut self, aabb: &Self) {
        self.insert_point(&aabb.min);
        self.insert_point(&aabb.max);
    }
}
type Bounds3f = Bound3<Float>;
impl Bound3<Float> {
    pub fn surface_area(&self) -> Float {
        let s = self.size();
        (s[0] * s[1] + s[1] * s[2] + s[0] * s[2]) * 2.0
    }
    pub fn centroid(&self) -> Vec3 {
        self.min + 0.5 * self.size()
    }
    pub fn diagonal(&self) -> Vec3 {
        self.max - self.min
    }
    pub fn contains(&self, p: &Vec3) -> bool {
        glm::all(&glm::less_than_equal(&p, &self.max))
            && glm::all(&glm::greater_than_equal(&p, &self.min))
    }
}
impl Default for Bound3<Float> {
    fn default() -> Self {
        let inf = Float::INFINITY;
        Self {
            min: vec3(inf, inf, inf),
            max: vec3(-inf, -inf, -inf),
        }
    }
}
impl<T> Bound3<T>
where
    T: glm::Number + na::ClosedDiv,
{
    pub fn offset(&self, p: &glm::TVec3<T>) -> glm::TVec3<T> {
        (p - self.min).component_div(&self.size())
    }
}

#[derive(Clone, Copy)]
pub struct Spectrum {
    pub samples: Vec3,
}
impl From<na::SVector<f32, { Spectrum::N_SAMPLES }>> for Spectrum {
    fn from(v: na::SVector<f32, { Spectrum::N_SAMPLES }>) -> Self {
        Spectrum { samples: v.cast() }
    }
}
impl Spectrum {
    pub const N_SAMPLES: usize = 3;
    pub fn from_srgb(rgb: &Vec3) -> Spectrum {
        let f = |s| -> Float {
            if s <= 0.04045 {
                s / 12.92
            } else {
                (((s + 0.055) / 1.055) as Float).powf(2.4)
            }
        };
        Spectrum {
            samples: vec3(f(rgb.x), f(rgb.y), f(rgb.z)),
        }
    }
    pub fn to_srgb(&self) -> Vec3 {
        let f = |l: Float| -> Float {
            if l <= 0.0031308 {
                l * 12.92
            } else {
                l.powf(1.0 / 2.4) * 1.055 - 0.055
            }
        };

        vec3(f(self.samples.x), f(self.samples.y), f(self.samples.z))
    }
    pub fn zero() -> Spectrum {
        Self {
            samples: glm::zero(),
        }
    }
    pub fn one() -> Spectrum {
        Self {
            samples: vec3(1.0, 1.0, 1.0),
        }
    }

    pub fn is_black(&self) -> bool {
        !glm::all(&glm::greater_than(&self.samples, &glm::zero()))
            || glm::any(&glm::less_than(&self.samples, &glm::zero()))
    }
}
impl Index<usize> for Spectrum {
    type Output = Float;
    fn index(&self, index: usize) -> &Self::Output {
        &self.samples[index]
    }
}
impl IndexMut<usize> for Spectrum {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.samples[index]
    }
}
impl std::ops::Add for Spectrum {
    type Output = Spectrum;
    fn add(self, rhs: Spectrum) -> Self::Output {
        Self {
            samples: self.samples + rhs.samples,
        }
    }
}
impl std::ops::AddAssign for Spectrum {
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}
impl std::ops::MulAssign for Spectrum {
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}
impl std::ops::MulAssign<Float> for Spectrum {
    fn mul_assign(&mut self, rhs: Float) {
        *self = *self * rhs;
    }
}
impl std::ops::Mul for Spectrum {
    type Output = Spectrum;
    fn mul(self, rhs: Spectrum) -> Self::Output {
        Self {
            samples: self.samples.component_mul(&rhs.samples),
        }
    }
}
impl std::ops::Mul<Float> for Spectrum {
    type Output = Spectrum;
    fn mul(self, rhs: Float) -> Self::Output {
        Self {
            samples: self.samples * rhs,
        }
    }
}
impl std::ops::Div<Float> for Spectrum {
    type Output = Spectrum;
    fn div(self, rhs: Float) -> Self::Output {
        Self {
            samples: self.samples / rhs,
        }
    }
}

#[derive(Clone, Copy)]
pub struct Ray {
    pub o: Vec3,
    pub d: Vec3,
    pub tmin: Float,
    pub tmax: Float,
    // pub time: Float,
}

impl Ray {
    pub fn spawn(o: &Vec3, d: &Vec3) -> Self {
        Self {
            o: *o,
            d: *d,
            tmin: 0.0,
            tmax: Float::INFINITY,
        }
    }
    pub fn offset_along_normal(&self, n: &Vec3) -> Self {
        Self {
            tmin: self.tmin + 0.001 / glm::dot(&self.d, &n),
            ..*self
        }
    }
    pub fn spawn_to(p1: &Vec3, p2: &Vec3) -> Self {
        let len = glm::length(&(p1 - p2));
        let mut ray = Self::spawn(&p1, &glm::normalize(&(p2 - p1)));
        ray.tmax = len;
        ray
    }
    pub fn at(&self, t: Float) -> Vec3 {
        self.o + t * self.d
    }
}
#[allow(non_snake_case)]
#[derive(Clone, Copy)]
pub struct Frame {
    pub N: Vec3,
    pub B: Vec3,
    pub T: Vec3,
}
impl Frame {
    pub fn from_normal(normal: &Vec3) -> Self {
        let tangent = if normal.x.abs() > normal.y.abs() {
            glm::normalize(&vec3(-normal.z, 0.0, normal.x))
        } else {
            glm::normalize(&vec3(0.0, normal.z, -normal.y))
        };
        Self {
            N: *normal,
            T: tangent,
            B: glm::normalize(&glm::cross(normal, &tangent)),
        }
    }
    pub fn to_local(&self, v: &Vec3) -> Vec3 {
        vec3(
            glm::dot(&v, &self.T),
            glm::dot(&v, &self.N),
            glm::dot(&v, &self.B),
        )
    }
    pub fn to_world(&self, v: &Vec3) -> Vec3 {
        self.T * v.x + self.N * v.y + self.B * v.z
    }
}
#[derive(Clone, Copy)]
struct Transform {
    m4: Mat4,
    inv_m4: Option<Mat4>,
    m3: Mat3,
    inv_m3: Option<Mat3>,
}
impl Transform {
    pub fn inverse(&self) -> Option<Transform> {
        Some(Self {
            m4: self.inv_m4?,
            inv_m4: Some(self.m4),
            m3: self.inv_m3?,
            inv_m3: Some(self.m3),
        })
    }
    pub fn identity() -> Self {
        Self {
            m4: glm::identity(),
            inv_m4: Some(glm::identity()),
            m3: glm::identity(),
            inv_m3: Some(glm::identity()),
        }
    }
    pub fn from_matrix(m: &Mat4) -> Self {
        let m3 = glm::mat4_to_mat3(&m);
        Self {
            m4: *m,
            inv_m4: m.try_inverse(),
            m3,
            inv_m3: m3.try_inverse(),
        }
    }
    pub fn transform_point(&self, p: &Vec3) -> Vec3 {
        let q = self.m4 * glm::TVec4::<Float>::new(p.x, p.y, p.z, 1.0);
        vec3(q.x, q.y, q.z) / q.w
    }
    pub fn transform_vector(&self, v: &Vec3) -> Vec3 {
        self.m3 * v
    }
    pub fn transform_normal(&self, n: &Vec3) -> Vec3 {
        self.inv_m3.unwrap().transpose() * n
    }
}
impl Mul for Transform {
    type Output = Transform;
    fn mul(self, rhs: Transform) -> Self::Output {
        Self {
            m4: self.m4 * rhs.m4,
            inv_m4: if let (Some(a), Some(b)) = (self.inv_m4, rhs.inv_m4) {
                Some(a * b)
            } else {
                None
            },
            m3: self.m3 * rhs.m3,
            inv_m3: if let (Some(a), Some(b)) = (self.inv_m3, rhs.inv_m3) {
                Some(a * b)
            } else {
                None
            },
        }
    }
}
const PI: Float = std::f64::consts::PI as Float;
const FRAC_1_PI: Float = std::f64::consts::FRAC_1_PI as Float;
const FRAC_PI_2: Float = std::f64::consts::FRAC_PI_2 as Float;
const FRAC_PI_4: Float = std::f64::consts::FRAC_PI_4 as Float;
pub fn concentric_sample_disk(u: &Vec2) -> Vec2 {
    let u_offset: Vec2 = 2.0 * u - vec2(1.0, 1.0);
    if u_offset.x == 0.0 && u_offset.y == 0.0 {
        return vec2(0.0, 0.0);
    }

    let (theta, r) = {
        if u_offset.x.abs() > u_offset.y.abs() {
            let r = u_offset.x;
            let theta = FRAC_PI_4 * (u_offset.y / u_offset.x);
            (theta, r)
        } else {
            let r = u_offset.y;
            let theta = FRAC_PI_2 - FRAC_PI_4 * (u_offset.x / u_offset.y);
            (theta, r)
        }
    };
    r * vec2(theta.cos(), theta.sin())
}
pub fn consine_hemisphere_sampling(u: &Vec2) -> Vec3 {
    let uv = concentric_sample_disk(&u);
    let r = glm::dot(&uv, &uv);
    let h = (1.0 - r).sqrt();
    vec3(uv.x, h, uv.y)
}
pub fn uniform_sphere_sampling(u: &Vec2) -> Vec3 {
    let z = 1.0 - 2.0 * u[0];
    let r = (1.0 - z * z).max(0.0).sqrt();
    let phi = 2.0 * PI * u[1];
    vec3(r * phi.cos(), z, r * phi.sin())
}
pub fn uniform_sphere_pdf() -> Float {
    1.0 / (4.0 * PI)
}
pub fn dir_to_spherical(v: &Vec3) -> Vec2 {
    let theta = v.y.acos();
    let phi = (v.z / v.x).atan() + PI;
    vec2(theta, phi)
}
pub fn parallel_for<F: Fn(usize) -> () + Sync>(count: usize, chunk_size: usize, f: F) {
    let chunks = (count + chunk_size - 1) / chunk_size;
    (0..chunks).into_par_iter().for_each(|chunk_id| {
        (chunk_id * chunk_size..(chunk_id * chunk_size + chunk_size).min(count)).for_each(|id| {
            f(id);
        });
    });
}
impl Frame {
    pub fn same_hemisphere(u: &Vec3, v: &Vec3) -> bool {
        u.y * v.y > 0.0
    }
    pub fn cos_theta(u: &Vec3) -> Float {
        u.y
    }
    pub fn abs_cos_theta(u: &Vec3) -> Float {
        u.y.abs()
    }
}
pub struct Intersection<'a> {
    pub shape: &'a dyn Shape,
    pub uv: Vec2,
    pub t: Float,
    pub ng: Vec3,
    pub ns: Vec3,
}

pub struct BSDFSample {
    pub wi: Vec3,
    pub f: Spectrum,
    pub pdf: Float,
}
#[derive(Clone, Copy)]
pub struct BSDFInfo {
    pub albedo: Spectrum,
    // pub roughness:Float,
}

pub trait BSDF: Sync + Send {
    fn evaluate(&self, wo: &Vec3, wi: &Vec3) -> Spectrum;
    fn evaluate_pdf(&self, wo: &Vec3, wi: &Vec3) -> Float;
    fn sample(&self, u: &Vec2, wo: &Vec3) -> Option<BSDFSample>;
    fn info(&self) -> BSDFInfo;
}
pub trait Shape: Sync + Send {
    fn intersect<'a>(&'a self, ray: &Ray) -> Option<Intersection<'a>>;
    fn occlude(&self, ray: &Ray) -> bool;
    fn bsdf<'a>(&'a self) -> Option<&'a dyn BSDF>;
    fn aabb(&self) -> Bounds3f;
}

pub trait Sampler: Sync + Send {
    fn next1d(&mut self) -> Float;
    fn next2d(&mut self) -> Vec2 {
        vec2(self.next1d(), self.next1d())
    }
}

pub trait Camera: Sync + Send {
    fn generate_ray(&self, pixel: &glm::UVec2, sampler: &mut dyn Sampler) -> (Ray, Spectrum);
    fn resolution(&self) -> glm::UVec2;
    fn we(&self, ray: &Ray) -> (Option<glm::UVec2>, Spectrum);
    fn pdf_we(&self, ray: &Ray) -> (Float, Float);
    fn n(&self) -> Vec3;
}
#[derive(Clone, Copy)]
pub struct LightRaySample {
    pub le: Spectrum,
    pub pdf_dir: Float,
    pub pdf_pos: Float,
    pub ray: Ray,
    pub n: Vec3,
}
#[derive(Clone, Copy)]
pub struct LightSample {
    pub li: Spectrum,
    pub pdf: Float,
    pub shadow_ray: Ray,
    pub wi: Vec3,
    pub p: Vec3,
}
#[derive(Clone, Copy)]
pub struct ReferencePoint {
    pub p: Vec3,
    pub n: Vec3,
}
bitflags! {
    pub struct LightFlags : u8 {
        const NONE = 0b0;
        const DELTA_POSITION = 0b1;
        const DELTA_DIRECTION = 0b10;
        const DELTA = Self::DELTA_POSITION.bits | Self::DELTA_DIRECTION.bits;
    }
}
pub trait Light: Sync + Send {
    fn sample_le(&self, u: &[Vec2; 2]) -> LightRaySample;
    fn sample_li(&self, u: &Vec2, p: &ReferencePoint) -> LightSample;
    fn pdf_le(&self, ray: &Ray) -> (Float, Float);
    fn pdf_li(&self, wi: &Vec3, p: &ReferencePoint) -> (Float, Float);
    fn le(&self, ray: &Ray) -> Spectrum;
    fn flags(&self) -> LightFlags;
    fn address(&self) -> usize; // ????
}
pub trait LightDistribution: Sync + Send {
    fn sample<'a>(&'a self, u: Float) -> (&'a dyn Light, Float);
    fn pdf<'a>(&self, light: &'a dyn Light) -> Float;
}
struct UniformLightDistribution {
    lights: Vec<Arc<dyn Light>>,
    pdf_map: HashMap<usize, Float>,
}
impl UniformLightDistribution {
    pub fn new(lights: Vec<Arc<dyn Light>>) -> Self {
        let mut pdf_map = HashMap::new();
        let pdf = 1.0 / lights.len() as Float;
        for i in &lights {
            pdf_map.insert(i.address(), pdf);
        }
        Self { lights, pdf_map }
    }
}
impl LightDistribution for UniformLightDistribution {
    fn sample<'a>(&'a self, u: Float) -> (&'a dyn Light, Float) {
        let idx = (u * self.lights.len() as Float) as usize;
        let pdf = 1.0 / self.lights.len() as Float;
        (self.lights[idx].as_ref(), pdf)
    }
    fn pdf<'a>(&self, light: &'a dyn Light) -> Float {
        if let Some(pdf) = self.pdf_map.get(&light.address()) {
            *pdf
        } else {
            0.0
        }
    }
}
#[derive(Copy, Clone)]
pub struct BSDFClosure<'a> {
    pub frame: Frame,
    pub bsdf: &'a dyn BSDF,
}
impl<'a> BSDFClosure<'a> {
    fn evaluate(&self, wo: &Vec3, wi: &Vec3) -> Spectrum {
        self.bsdf
            .evaluate(&self.frame.to_local(&wo), &self.frame.to_local(&wi))
    }
    fn evaluate_pdf(&self, wo: &Vec3, wi: &Vec3) -> Float {
        self.bsdf
            .evaluate_pdf(&self.frame.to_local(&wo), &self.frame.to_local(&wi))
    }
    fn sample(&self, u: &Vec2, wo: &Vec3) -> Option<BSDFSample> {
        let mut sample = self.bsdf.sample(&u, &self.frame.to_local(&wo))?;
        sample.wi = self.frame.to_world(&sample.wi);
        Some(sample)
    }
}
#[derive(Copy, Clone)]
struct Pixel {
    intensity: Spectrum,
    weight: Float,
}
struct Film {
    pixels: RwLock<Vec<Pixel>>,
    resolution: glm::UVec2,
}
impl Film {
    pub fn new(resolution: &glm::UVec2) -> Self {
        Self {
            pixels: RwLock::new(vec![
                Pixel {
                    intensity: Spectrum::zero(),
                    weight: 0.0
                };
                (resolution.x * resolution.y) as usize
            ]),
            resolution: *resolution,
        }
    }
    pub fn add_sample(&self, pixel: &glm::UVec2, value: &Spectrum, weight: Float) {
        let mut pixels = self.pixels.write().unwrap();
        let pixel = &mut (*pixels)[(pixel.x + pixel.y * self.resolution.x) as usize];
        pixel.intensity = pixel.intensity + *value;
        pixel.weight += weight;
    }
    pub fn get_pixel(&self, pixel: &glm::UVec2) -> Pixel {
        let pixels = self.pixels.read().unwrap();
        (*pixels)[(pixel.x + pixel.y * self.resolution.x) as usize]
    }
    pub fn to_rgb_image(&self) -> image::RgbImage {
        let image = image::ImageBuffer::from_fn(self.resolution.x, self.resolution.y, |x, y| {
            let pixel = self.get_pixel(&uvec2(x, y));
            let value = pixel.intensity * (1.0 / pixel.weight);
            let srgb = value.to_srgb() * 255.0;
            image::Rgb([srgb.x as u8, srgb.y as u8, srgb.z as u8])
        });

        image
    }
}
mod bvh {
    use std::mem::swap;

    use super::*;
    #[derive(Clone, Copy, Debug)]
    pub struct BVHNode {
        pub axis: u8,
        pub aabb: Bounds3f,
        pub first: u32,
        pub count: u32,
        pub left: u32,
        pub right: u32,
    }
    impl BVHNode {
        pub fn is_leaf(&self) -> bool {
            self.count > 0
        }
    }
    #[derive(Clone)]
    pub struct BVHAccelerator<T: Shape + Clone> {
        pub primitives: Vec<T>,
        pub nodes: Vec<BVHNode>,
        pub aabb: Bounds3f,
    }
    #[derive(Default, Clone, Copy)]
    struct Bucket {
        count: usize,
        aabb: Bounds3f,
    }
    impl<T> BVHAccelerator<T>
    where
        T: Shape + Clone,
    {
        pub fn new(primitives: Vec<T>) -> Self {
            let mut bvh = Self {
                primitives,
                nodes: vec![],
                aabb: Bounds3f::default(),
            };
            bvh.recursive_build(0, bvh.primitives.len() as u32, 0);
            println!("bvh nodes: {}", bvh.nodes.len());
            // for node in &bvh.nodes {
            //     println!("{:?}", node);
            // }
            bvh
        }
        fn recursive_build(&mut self, begin: u32, end: u32, depth: u32) -> u32 {
            // println!("building {}..{}", begin, end);
            let mut aabb = Bounds3f::default();
            for i in begin..end {
                aabb.insert_box(&self.primitives[i as usize].aabb());
            }
            if depth == 0 {
                self.aabb = aabb;
            }
            if end - begin <= 4 || depth >= 28 {
                if end - begin == 0 {
                    panic!("");
                }
                let node = BVHNode {
                    axis: 0,
                    aabb,
                    first: begin,
                    count: end - begin,
                    left: 0,
                    right: 0,
                };
                self.nodes.push(node);
                return (self.nodes.len() - 1) as u32;
            } else {
                let size = aabb.size();
                let axis = {
                    if size[0] > size[1] && size[0] > size[2] {
                        0
                    } else if size[1] > size[0] && size[1] > size[2] {
                        1
                    } else {
                        2
                    }
                };
                const N_BUCKETS: usize = 12;
                let mut buckets = [Bucket::default(); N_BUCKETS];
                for i in begin..end {
                    let p_aabb = self.primitives[i as usize].aabb();
                    let b = (N_BUCKETS - 1)
                        .min((aabb.offset(&p_aabb.centroid())[axis] * N_BUCKETS as Float) as usize);
                    buckets[b].count += 1;
                    buckets[b].aabb.insert_box(&p_aabb);
                }
                let mut costs = [0.0 as Float; N_BUCKETS - 1];
                for i in 0..N_BUCKETS - 1 {
                    let mut b0 = Bounds3f::default();
                    let mut b1 = Bounds3f::default();
                    let mut count0 = 0;
                    let mut count1 = 0;
                    for j in 0..=i {
                        b0.insert_box(&buckets[j].aabb);
                        count0 += buckets[j].count;
                    }
                    for j in i + 1..N_BUCKETS {
                        b1.insert_box(&buckets[j].aabb);
                        count1 += buckets[j].count;
                    }
                    costs[i] = 0.125
                        + (count0 as Float * b0.surface_area())
                        + (count1 as Float * b1.surface_area());
                }
                let mut split_buckets = 0;
                let mut min_cost = Float::INFINITY;
                for i in 0..N_BUCKETS - 1 {
                    if costs[i] < min_cost {
                        min_cost = costs[i];
                        split_buckets = i;
                    }
                }
                // partition
                {
                    let predicate = |shape: &T| {
                        let b = {
                            let b = (aabb.offset(&shape.aabb().centroid())[axis]
                                * N_BUCKETS as Float) as usize;
                            b.min(N_BUCKETS - 1)
                        };
                        b <= split_buckets
                    };
                    let mut first = (|| {
                        for i in begin..end {
                            if !predicate(&self.primitives[i as usize]) {
                                return i;
                            }
                        }
                        end
                    })();
                    let mut mid: u32 = (|| {
                        if first == end {
                            return first;
                        }
                        for i in first + 1..end {
                            if predicate(&self.primitives[i as usize]) {
                                self.primitives.swap(first as usize, i as usize);
                                first += 1;
                            }
                        }
                        return first;
                    })();
                    if mid == begin || mid == end {
                        if end - begin > 12 {
                            eprintln!(
                                "cannot split at depth {} with {} primitives",
                                depth,
                                end - begin
                            );
                        }
                        mid = (end + begin) / 2;
                    }
                    let ret = self.nodes.len();
                    self.nodes.push(BVHNode {
                        axis: axis as u8,
                        aabb,
                        first: 0,
                        count: 0,
                        left: 0,
                        right: 0,
                    });
                    self.nodes[ret].left = self.recursive_build(begin, mid, depth + 1);
                    self.nodes[ret].right = self.recursive_build(mid, end, depth + 1);
                    return ret as u32;
                }
            }
        }
        fn intersect_aabb(aabb: &Bounds3f, ray: &Ray, invd: &Vec3) -> Float {
            let t0 = (aabb.min - ray.o).component_mul(&invd);
            let t1 = (aabb.max - ray.o).component_mul(&invd);
            let min = glm::min2(&t0, &t1);
            let max = glm::max2(&t0, &t1);
            let tmin = glm::comp_max(&min).max(ray.tmin);
            let tmax = glm::comp_min(&max).min(ray.tmax);
            if tmin <= tmax {
                tmin
            } else {
                -1.0
            }
        }
        fn intersect_leaf<'a>(&'a self, node: &BVHNode, ray: &mut Ray) -> Option<Intersection<'a>> {
            let first = node.first;
            let last = node.first + node.count;
            let mut ret = None;
            for i in first..last {
                if let Some(isct) = self.primitives[i as usize].intersect(&ray) {
                    ray.tmax = isct.t;
                    ret = Some(isct);
                }
            }
            ret
        }
        pub fn intersect<'a>(&'a self, original_ray: &Ray) -> Option<Intersection<'a>> {
            let mut stack: [Option<&BVHNode>; 64] = [None; 64];
            let mut sp = 0;
            let mut p = Some(&self.nodes[0]);
            let mut ray = *original_ray;
            let invd: Vec3 = vec3(1.0, 1.0, 1.0).component_div(&ray.d);
            let mut isct = None;
            while p.is_some() {
                let node = p.unwrap();
                let t = Self::intersect_aabb(&node.aabb, &ray, &invd);
                if t < 0.0 {
                    if sp > 0 {
                        sp -= 1;
                        p = stack[sp];
                    } else {
                        p = None;
                    }
                    continue;
                }
                if node.is_leaf() {
                    if let Some(hit) = self.intersect_leaf(node, &mut ray) {
                        isct = Some(hit);
                    }
                    if sp > 0 {
                        sp -= 1;
                        p = stack[sp];
                    } else {
                        p = None;
                    }
                } else {
                    if ray.d[node.axis as usize] > 0.0 {
                        stack[sp] = Some(&self.nodes[node.right as usize]);
                        sp += 1;
                        p = Some(&self.nodes[node.left as usize]);
                    } else {
                        stack[sp] = Some(&self.nodes[node.left as usize]);
                        sp += 1;
                        p = Some(&self.nodes[node.right as usize]);
                    }
                }
            }
            isct
        }
        pub fn occlude(&self, original_ray: &Ray) -> bool {
            let mut stack: [Option<&BVHNode>; 64] = [None; 64];
            let mut sp = 0;
            let mut p = Some(&self.nodes[0]);
            let mut ray = *original_ray;
            let invd: Vec3 = vec3(1.0, 1.0, 1.0).component_div(&ray.d);
            while p.is_some() {
                let node = p.unwrap();
                let t = Self::intersect_aabb(&node.aabb, &ray, &invd);
                if t < 0.0 {
                    if sp > 0 {
                        sp -= 1;
                        p = stack[sp];
                    } else {
                        p = None;
                    }
                    continue;
                }
                if node.is_leaf() {
                    if let Some(_) = self.intersect_leaf(node, &mut ray) {
                        return true;
                    }
                    if sp > 0 {
                        sp -= 1;
                        p = stack[sp];
                    } else {
                        p = None;
                    }
                } else {
                    if ray.d[node.axis as usize] > 0.0 {
                        stack[sp] = Some(&self.nodes[node.right as usize]);
                        sp += 1;
                        p = Some(&self.nodes[node.left as usize]);
                    } else {
                        stack[sp] = Some(&self.nodes[node.left as usize]);
                        sp += 1;
                        p = Some(&self.nodes[node.right as usize]);
                    }
                }
            }
            false
        }
    }
}
impl Shape for Arc<dyn Shape> {
    fn intersect<'a>(&'a self, ray: &Ray) -> Option<Intersection<'a>> {
        self.as_ref().intersect(ray)
    }
    fn occlude(&self, ray: &Ray) -> bool {
        self.as_ref().occlude(ray)
    }
    fn bsdf<'a>(&'a self) -> Option<&'a dyn BSDF> {
        self.as_ref().bsdf()
    }
    fn aabb(&self) -> Bounds3f {
        self.as_ref().aabb()
    }
}
#[derive(Clone)]
struct Aggregate {
    bvh: bvh::BVHAccelerator<Arc<dyn Shape>>,
}
impl Aggregate {
    pub fn new(shapes: Vec<Arc<dyn Shape>>) -> Self {
        Self {
            bvh: bvh::BVHAccelerator::new(shapes),
        }
    }
}
impl Shape for Aggregate {
    fn intersect<'a>(&'a self, original_ray: &Ray) -> Option<Intersection<'a>> {
        self.bvh.intersect(original_ray)
    }
    fn occlude(&self, ray: &Ray) -> bool {
        self.bvh.occlude(ray)
    }
    fn bsdf<'a>(&'a self) -> Option<&'a dyn BSDF> {
        None
    }
    fn aabb(&self) -> Bounds3f {
        Bounds3f::default()
    }
}
#[derive(Clone)]
struct Sphere {
    center: Vec3,
    radius: Float,
    bsdf: Arc<dyn BSDF>,
}

impl Shape for Sphere {
    fn intersect<'a>(&'a self, ray: &Ray) -> Option<Intersection<'a>> {
        let oc = ray.o - self.center;
        let a = glm::dot(&ray.d, &ray.d);
        let b = 2.0 * glm::dot(&ray.d, &oc);
        let c = glm::dot(&oc, &oc) - self.radius * self.radius;
        let delta = b * b - 4.0 * a * c;
        if delta < 0.0 {
            return None;
        }
        let t1 = (-b - delta.sqrt()) / (2.0 * a);
        if t1 >= ray.tmin && t1 < ray.tmax {
            let p = ray.at(t1);
            let n = glm::normalize(&(p - self.center));
            return Some(Intersection {
                shape: self,
                t: t1,
                uv: vec2(0.0, 0.0),
                ng: n,
                ns: n,
            });
        }
        let t2 = (-b + delta.sqrt()) / (2.0 * a);
        if t2 >= ray.tmin && t2 < ray.tmax {
            let p = ray.at(t2);
            let n = glm::normalize(&(p - self.center));
            return Some(Intersection {
                shape: self,
                t: t2,
                uv: vec2(0.0, 0.0),
                ng: n,
                ns: n,
            });
        }
        None
    }
    fn occlude(&self, ray: &Ray) -> bool {
        let oc = ray.o - self.center;
        let a = glm::dot(&ray.d, &ray.d);
        let b = 2.0 * glm::dot(&ray.d, &oc);
        let c = glm::dot(&oc, &oc) - self.radius * self.radius;
        let delta = b * b - 4.0 * a * c;
        if delta < 0.0 {
            return false;
        }
        let t1 = (-b - delta.sqrt()) / (2.0 * a);
        if t1 >= ray.tmin && t1 < ray.tmax {
            return true;
        }
        let t2 = (-b + delta.sqrt()) / (2.0 * a);
        if t2 >= ray.tmin && t2 < ray.tmax {
            return true;
        }
        false
    }
    fn bsdf<'a>(&'a self) -> Option<&'a dyn BSDF> {
        Some(&*self.bsdf)
    }
    fn aabb(&self) -> Bounds3f {
        Bounds3f {
            min: self.center - vec3(self.radius, self.radius, self.radius),
            max: self.center + vec3(self.radius, self.radius, self.radius),
        }
    }
}

#[derive(Clone)]
struct TriangleMesh {
    vertices: Vec<Vec3>,
    indices: Vec<u32>,
}

struct PerspectiveCamera {
    resolution: glm::UVec2,
    c2w: Transform,
    w2c: Transform,
    fov: Float,
    r2c: Transform,
    c2r: Transform,
    a: Float,
}
impl PerspectiveCamera {
    fn new(resolution: &glm::UVec2, transform: &Transform, fov: Float) -> Self {
        let mut m = glm::identity();
        let fres = vec2(resolution.x as Float, resolution.y as Float);
        m = glm::scale(&glm::identity(), &vec3(1.0 / fres.x, 1.0 / fres.y, 1.0)) * m;
        m = glm::scale(&glm::identity(), &vec3(2.0, 2.0, 1.0)) * m;
        m = glm::translate(&glm::identity(), &vec3(-1.0, -1.0, 0.0)) * m;
        m = glm::scale(&glm::identity(), &vec3(1.0, -1.0, 1.0)) * m;
        let s = (fov / 2.0).atan();
        if resolution.x > resolution.y {
            m = glm::scale(&glm::identity(), &vec3(s, s * fres.y / fres.x, 1.0)) * m;
        } else {
            m = glm::scale(&glm::identity(), &vec3(s * fres.x / fres.y, s, 1.0)) * m;
        }
        let r2c = Transform::from_matrix(&m);
        let a = {
            let p_min = r2c.transform_point(&vec3(0.0, 0.0, 0.0));
            let p_max =
                r2c.transform_point(&vec3(resolution.x as Float, resolution.y as Float, 0.0));
            ((p_max.x - p_min.x) * (p_max.y * p_min.y)).abs()
        };
        Self {
            resolution: *resolution,
            c2w: *transform,
            w2c: transform.inverse().unwrap(),
            r2c,
            c2r: r2c.inverse().unwrap(),
            fov,
            a,
        }
    }
}
impl Camera for PerspectiveCamera {
    fn generate_ray(&self, pixel: &glm::UVec2, sampler: &mut dyn Sampler) -> (Ray, Spectrum) {
        // let p_lens = consine_hemisphere_sampling(&sampler.next2d())
        let fpixel: Vec2 = glm::convert(*pixel);
        let p_film = sampler.next2d() + fpixel;
        let p = {
            let v = self.r2c.transform_point(&vec3(p_film.x, p_film.y, 0.0));
            vec2(v.x, v.y)
        };
        let mut ray = Ray::spawn(
            &glm::zero(),
            &glm::normalize(&(vec3(p.x, p.y, 0.0) - vec3(0.0, 0.0, 1.0))),
        );
        // ray.tmin = (1.0 / ray.d.z).abs();
        ray.o = self.c2w.transform_point(&ray.o);
        ray.d = self.c2w.transform_vector(&ray.d);
        (ray, Spectrum::one())
    }
    fn resolution(&self) -> glm::UVec2 {
        self.resolution
    }
    fn we(&self, ray: &Ray) -> (Option<glm::UVec2>, Spectrum) {
        let cos_theta = glm::dot(&ray.d, &self.c2w.transform_vector(&vec3(0.0, 0.0, -1.0)));
        if cos_theta <= 0.0 {
            return (None, Spectrum::zero());
        }
        let p_focus = ray.at(1.0 / cos_theta);
        let p_raster = self
            .c2r
            .transform_point(&self.w2c.transform_point(&p_focus));
        if p_raster.x < 0.0
            || p_raster.x >= self.resolution().x as Float
            || p_raster.y < 0.0
            || p_raster.y >= self.resolution().y as Float
        {
            return (None, Spectrum::zero());
        }
        let lens_area = 1.0;
        let cos2_theta = cos_theta * cos_theta;
        (
            Some(uvec2(p_raster.x as u32, p_raster.y as u32)),
            Spectrum::one() / (self.a * lens_area * cos2_theta * cos2_theta),
        )
    }
    fn pdf_we(&self, ray: &Ray) -> (Float, Float) {
        let cos_theta = glm::dot(&ray.d, &self.c2w.transform_vector(&vec3(0.0, 0.0, -1.0)));
        if cos_theta <= 0.0 {
            return (0.0, 0.0);
        }
        let p_focus = ray.at(1.0 / cos_theta);
        let p_raster = self
            .c2r
            .transform_point(&self.w2c.transform_point(&p_focus));
        if p_raster.x < 0.0
            || p_raster.x >= self.resolution().x as Float
            || p_raster.y < 0.0
            || p_raster.y >= self.resolution().y as Float
        {
            return (0.0, 0.0);
        }
        let lens_area = 1.0;
        let cos2_theta = cos_theta * cos_theta;
        (1.0 / lens_area, self.a * cos2_theta * cos_theta)
    }
    fn n(&self) -> Vec3 {
        self.c2w.transform_vector(&vec3(0.0, 0.0, -1.0))
    }
}
struct NullBSDF {}
impl BSDF for NullBSDF {
    fn evaluate(&self, wo: &Vec3, wi: &Vec3) -> Spectrum {
        Spectrum::zero()
    }
    fn evaluate_pdf(&self, wo: &Vec3, wi: &Vec3) -> Float {
        0.0
    }
    fn sample(&self, u: &Vec2, wo: &Vec3) -> Option<BSDFSample> {
        None
    }
    fn info(&self) -> BSDFInfo {
        BSDFInfo {
            albedo: Spectrum::zero(),
        }
    }
}
struct PCG {
    state: usize,
}
impl PCG {
    const MULTIPILER: usize = 6364136223846793005;
    const INC: usize = 1442695040888963407;
    pub fn pcg32(&mut self) -> u32 {
        let mut x = self.state;
        let count = x >> 59;
        self.state = x.wrapping_mul(Self::MULTIPILER).wrapping_add(Self::INC); //x * Self::MULTIPILER + Self::INC;
        x ^= x >> 18;
        ((x >> 27) as u32).rotate_right(count as u32)
    }
    pub fn new(seed: usize) -> Self {
        let mut r = Self {
            state: seed + Self::INC,
        };
        let _ = r.pcg32();
        r
    }
}
struct PCGSampler {
    rng: PCG,
}
impl Sampler for PCGSampler {
    fn next1d(&mut self) -> Float {
        self.rng.pcg32() as Float / (std::u32::MAX as Float)
    }
}
struct PointLight {
    position: Vec3,
    emission: Spectrum,
}
impl Light for PointLight {
    fn sample_le(&self, u: &[Vec2; 2]) -> LightRaySample {
        let w = uniform_sphere_sampling(&u[0]);
        LightRaySample {
            le: self.emission,
            pdf_pos: 1.0,
            pdf_dir: uniform_sphere_pdf(),
            ray: Ray::spawn(&self.position, &w),
            n: w,
        }
    }
    fn sample_li(&self, _: &Vec2, ref_: &ReferencePoint) -> LightSample {
        let mut ray = Ray::spawn_to(&self.position, &ref_.p);
        let len2 = {
            let v = self.position - ref_.p;
            glm::dot(&v, &v)
        };
        ray.tmax *= 0.997;
        LightSample {
            li: self.emission / len2,
            pdf: 1.0,
            shadow_ray: ray,
            wi: glm::normalize(&(self.position - ref_.p)),
            p: self.position,
        }
    }
    fn pdf_le(&self, ray: &Ray) -> (Float, Float) {
        (0.0, uniform_sphere_pdf())
    }
    fn pdf_li(&self, wi: &Vec3, p: &ReferencePoint) -> (Float, Float) {
        (0.0, 0.0)
    }
    fn le(&self, _: &Ray) -> Spectrum {
        Spectrum::zero()
        // unimplemented!("point light cannot be hit")
    }
    fn flags(&self) -> LightFlags {
        LightFlags::DELTA_POSITION
    }
    fn address(&self) -> usize {
        self as *const PointLight as usize
    }
}
struct DiffuseBSDF {
    reflecance: Spectrum,
}
impl BSDF for DiffuseBSDF {
    fn info(&self) -> BSDFInfo {
        BSDFInfo {
            albedo: self.reflecance,
        }
    }
    fn evaluate(&self, wo: &Vec3, wi: &Vec3) -> Spectrum {
        if Frame::same_hemisphere(&wo, &wi) {
            self.reflecance * FRAC_1_PI
        } else {
            Spectrum::zero()
        }
    }
    fn evaluate_pdf(&self, wo: &Vec3, wi: &Vec3) -> Float {
        if Frame::same_hemisphere(&wo, &wi) {
            Frame::abs_cos_theta(&wi) * FRAC_1_PI
        } else {
            0.0
        }
    }
    fn sample(&self, u: &Vec2, wo: &Vec3) -> Option<BSDFSample> {
        let wi = {
            let w = consine_hemisphere_sampling(&u);
            if Frame::same_hemisphere(&w, &wo) {
                w
            } else {
                vec3(w.x, -w.y, w.z)
            }
        };
        Some(BSDFSample {
            f: self.reflecance * FRAC_1_PI,
            wi,
            pdf: Frame::abs_cos_theta(&wi) * FRAC_1_PI,
        })
    }
}
pub struct Scene {
    pub shape: Arc<dyn Shape>,
    pub camera: Arc<dyn Camera>,
    pub lights: Vec<Arc<dyn Light>>,
    pub light_distr: Arc<dyn LightDistribution>,
}

trait Integrator {
    type Output;
    fn render(&mut self, scene: &Scene) -> Self::Output;
}
struct RTAO {
    spp: u32,
}
impl Integrator for RTAO {
    type Output = Film;
    fn render(&mut self, scene: &Scene) -> Self::Output {
        let npixels = (scene.camera.resolution().x * scene.camera.resolution().y) as usize;
        let film = RwLock::new(Film::new(&scene.camera.resolution()));
        parallel_for(npixels, 256, |id| {
            let mut sampler = PCGSampler { rng: PCG::new(id) };
            let x = (id as u32) % scene.camera.resolution().x;
            let y = (id as u32) / scene.camera.resolution().x;
            let pixel = uvec2(x, y);
            let mut acc_li = Spectrum::zero();
            for _ in 0..self.spp {
                let (mut ray, ray_weight) = scene.camera.generate_ray(&pixel, &mut sampler);
                let mut li = Spectrum::zero();
                {
                    if let Some(isct) = scene.shape.intersect(&ray) {
                        // li = Spectrum { samples: isct.ng }
                        let ng = isct.ng;
                        let frame = Frame::from_normal(&ng);
                        let wi = {
                            let w = consine_hemisphere_sampling(&sampler.next2d());
                            frame.to_world(&w)
                        };
                        // li = Spectrum {
                        //     samples: vec3(1.0,1.0,1.0) * glm::dot(&wi, &vec3(0.2, 0.8, 0.0)),
                        // };

                        // li = Spectrum{samples:wi};
                        let p = ray.at(isct.t);
                        ray = Ray::spawn(&p, &wi).offset_along_normal(&ng);
                        if !scene.shape.occlude(&ray) {
                            li = Spectrum::one();
                        }
                    }
                }
                acc_li = acc_li + li;
            }
            acc_li = acc_li / (self.spp as Float);
            {
                let film = &mut film.write().unwrap();
                film.add_sample(&uvec2(x, y), &acc_li, 1.0);
            }
        });
        film.into_inner().unwrap()
    }
}
struct PathTracer {
    spp: u32,
    max_depth: u32,
}
impl Integrator for PathTracer {
    type Output = Film;
    fn render(&mut self, scene: &Scene) -> Self::Output {
        let npixels = (scene.camera.resolution().x * scene.camera.resolution().y) as usize;
        let film = Film::new(&scene.camera.resolution());
        parallel_for(npixels, 256, |id| {
            let mut sampler = PCGSampler { rng: PCG::new(id) };
            let x = (id as u32) % scene.camera.resolution().x;
            let y = (id as u32) / scene.camera.resolution().x;
            let pixel = uvec2(x, y);
            let mut acc_li = Spectrum::zero();
            for _ in 0..self.spp {
                let (mut ray, ray_weight) = scene.camera.generate_ray(&pixel, &mut sampler);
                let mut li = Spectrum::zero();
                let mut beta = Spectrum::one();
                {
                    let mut depth = 0;
                    loop {
                        if let Some(isct) = scene.shape.intersect(&ray) {
                            let ng = isct.ng;
                            let frame = Frame::from_normal(&ng);
                            let shape = isct.shape;
                            let opt_bsdf = shape.bsdf();
                            if opt_bsdf.is_none() {
                                break;
                            }
                            let p = ray.at(isct.t);
                            let bsdf = BSDFClosure {
                                frame,
                                bsdf: opt_bsdf.unwrap(),
                            };
                            let wo = -ray.d;
                            if depth >= self.max_depth {
                                break;
                            }
                            depth += 1;
                            {
                                let (light, light_pdf) = scene.light_distr.sample(sampler.next1d());
                                let p_ref = ReferencePoint { p, n: ng };
                                let light_sample = light.sample_li(&sampler.next2d(), &p_ref);
                                if !light_sample.li.is_black()
                                    && !scene.shape.occlude(&light_sample.shadow_ray)
                                {
                                    li += beta
                                        * bsdf.evaluate(&wo, &light_sample.wi)
                                        * glm::dot(&ng, &light_sample.wi).abs()
                                        * light_sample.li
                                        / (light_sample.pdf * light_pdf);
                                }
                            }

                            if let Some(bsdf_sample) = bsdf.sample(&sampler.next2d(), &wo) {
                                let wi = &bsdf_sample.wi;
                                ray = Ray::spawn(&p, wi).offset_along_normal(&ng);
                                beta *= bsdf_sample.f * glm::dot(wi, &ng).abs() / bsdf_sample.pdf;
                            } else {
                                break;
                            }
                        } else {
                            break;
                        }
                    }
                }
                acc_li += li;
            }
            acc_li = acc_li / (self.spp as Float);
            film.add_sample(&uvec2(x, y), &acc_li, 1.0);
        });
        film
    }
}

#[derive(Clone)]
struct VisiblePoint<'a> {
    pub bsdf: BSDFClosure<'a>,
    pub p: Vec3,
    pub wo: Vec3,
    pub beta: Spectrum,
}

#[allow(non_snake_case)]
struct SPPMPixel<'a> {
    radius: Float,
    ld: Spectrum,
    M: AtomicU32,
    N: Float,
    phi: [AtomicFloat; Spectrum::N_SAMPLES],
    tau: Spectrum,
    vp: Option<VisiblePoint<'a>>,
}

impl<'a> Clone for SPPMPixel<'a> {
    fn clone(&self) -> Self {
        Self {
            radius: self.radius,
            ld: self.ld,
            M: AtomicU32::new(self.M.load(Ordering::Relaxed)),
            phi: self.phi.clone(),
            N: self.N,
            tau: self.tau,
            vp: self.vp.clone(),
        }
    }
}
struct SPPMPixelListNode<'a> {
    pixel: &'a SPPMPixel<'a>,
    next: AtomicPtr<SPPMPixelListNode<'a>>,
}
// impl<'a> Drop for SPPMPixelListNode<'a> {
//     fn drop(&mut self) {
//         unsafe {
//             let p = self.next.load(Ordering::Relaxed);
//             if !p.is_null() {
//                 Box::from_raw(p);
//             }
//         }
//     }
// }

struct SPPMPixelList<'a>(AtomicPtr<SPPMPixelListNode<'a>>);
impl<'a> Clone for SPPMPixelList<'a> {
    fn clone(&self) -> Self {
        Self(AtomicPtr::new(self.0.load(Ordering::SeqCst)))
    }
}
struct VisiblePointGrid<'a> {
    bound: Bounds3f,
    grid: Vec<SPPMPixelList<'a>>,
    hash_size: usize,
    grid_res: [u32; 3],
}
impl<'a> VisiblePointGrid<'a> {
    fn hash(&self, p: &glm::UVec3) -> usize {
        ((p.x as usize * 73856093) ^ (p.y as usize * 19349663) ^ (p.z as usize * 83492791))
            % self.hash_size
    }
    pub fn new(bound: &Bounds3f, grid_res: [u32; 3], hash_size: usize) -> Self {
        Self {
            bound: *bound,
            grid: vec![SPPMPixelList(AtomicPtr::new(std::ptr::null_mut())); hash_size],
            hash_size,
            grid_res,
        }
    }
    pub fn to_grid(&self, mut p: Vec3) -> glm::UVec3 {
        p = glm::min2(&self.bound.max, &p);
        p = glm::max2(&self.bound.min, &p);
        let mut q = self.bound.offset(&p);
        q = q.component_mul(&vec3(
            self.grid_res[0] as Float,
            self.grid_res[1] as Float,
            self.grid_res[2] as Float,
        ));
        uvec3(q.x as u32, q.y as u32, q.z as u32)
    }
    pub fn insert(&self, pixel: &'a SPPMPixel<'a>) {
        let p = pixel.vp.as_ref().unwrap().p;
        let radius = pixel.radius;
        let pmin = self.to_grid(p - vec3(radius, radius, radius));
        let pmax = self.to_grid(p + vec3(radius, radius, radius));
        // println!("{:?} {:?}", pmin,pmax);
        for z in pmin.z..=pmax.z {
            for y in pmin.y..=pmax.y {
                for x in pmin.x..=pmax.x {
                    let h = self.hash(&uvec3(x, y, z));

                    self.insert_at(h, pixel);
                }
            }
        }
    }
    fn insert_at(&self, h: usize, pixel: &'a SPPMPixel<'a>) {
        // let p = pixel.vp.as_ref().unwrap().p;
        // let h = self.hash(&self.to_grid(p));
        let ap = &self.grid[h].0;
        let node: *mut SPPMPixelListNode<'a> = Box::into_raw(Box::new(SPPMPixelListNode {
            pixel,
            next: AtomicPtr::new(ap.load(Ordering::SeqCst)),
        }));
        loop {
            unsafe {
                match ap.compare_exchange_weak(
                    (*node).next.load(Ordering::SeqCst),
                    node,
                    Ordering::SeqCst,
                    Ordering::SeqCst,
                ) {
                    Ok(_) => break,
                    Err(x) => (*node).next.store(x, Ordering::SeqCst),
                }
            }
        }
    }
}

#[derive(Clone, Copy)]
pub struct UnsafePointer<T> {
    p: *mut T,
}
unsafe impl<T> Sync for UnsafePointer<T> {}
unsafe impl<T> Send for UnsafePointer<T> {}
impl<T> UnsafePointer<T> {
    fn new(p: *mut T) -> Self {
        Self { p }
    }
    unsafe fn as_mut<'a>(&self) -> Option<&'a mut T> {
        self.p.as_mut()
    }
    unsafe fn as_ref<'a>(&self) -> Option<&'a T> {
        self.p.as_ref()
    }
    unsafe fn offset(&self, count: isize) -> Self {
        Self {
            p: self.p.offset(count),
        }
    }
}
impl<'a> Drop for VisiblePointGrid<'a> {
    fn drop(&mut self) {
        for p in &self.grid {
            let mut p = p.0.load(Ordering::Relaxed);
            while !p.is_null() {
                unsafe {
                    let q = p;
                    p = (*p).next.load(Ordering::Relaxed);
                    Box::from_raw(q);
                }
            }
        }
    }
}
struct SPPM {
    iterations: usize,
    max_depth: usize,
    initial_radius: Float,
    n_photons: usize,
}

impl Integrator for SPPM {
    type Output = Film;
    fn render(&mut self, scene: &Scene) -> Self::Output {
        let npixels = (scene.camera.resolution().x * scene.camera.resolution().y) as usize;
        let film = Film::new(&scene.camera.resolution());
        let mut pixels: Vec<SPPMPixel> = vec![
            SPPMPixel {
                radius: self.initial_radius,
                ld: Spectrum::zero(),
                N: 0.0,
                M: AtomicU32::new(0),
                tau: Spectrum::zero(),
                vp: None,
                phi: Default::default(),
            };
            npixels
        ];
        let mut samplers: Vec<Box<dyn Sampler>> = vec![];
        for i in 0..npixels {
            samplers.push(Box::new(PCGSampler { rng: PCG::new(i) }));
        }
        let mut photon_samplers: Vec<Box<dyn Sampler>> = vec![];
        for i in 0..self.n_photons {
            photon_samplers.push(Box::new(PCGSampler { rng: PCG::new(i) }));
        }
        #[allow(unused_assignments)]
        let mut grid: Option<VisiblePointGrid> = None;

        let p_samplers = &UnsafePointer::new(samplers.as_mut_ptr());
        let p_pixels = &UnsafePointer::new(pixels.as_mut_ptr());
        let p_photon_samplers = &UnsafePointer::new(photon_samplers.as_mut_ptr());
        for _iteration in 0..self.iterations {
            parallel_for(npixels, 256, |id| {
                let sppm_pixel = unsafe { p_pixels.offset(id as isize).as_mut().unwrap() };
                sppm_pixel.vp = None;
                sppm_pixel.M.store(0, Ordering::Relaxed);

                let x = (id as u32) % scene.camera.resolution().x;
                let y = (id as u32) / scene.camera.resolution().x;
                let pixel = uvec2(x, y);

                let sampler = unsafe { p_samplers.offset(id as isize).as_mut().unwrap().as_mut() };
                let (mut ray, ray_weight) = scene.camera.generate_ray(&pixel, sampler);
                if let Some(isct) = scene.shape.intersect(&ray) {
                    let ng = isct.ng;
                    let frame = Frame::from_normal(&ng);
                    let shape = isct.shape;
                    let opt_bsdf = shape.bsdf();
                    if opt_bsdf.is_none() {
                        return;
                    }
                    let p = ray.at(isct.t);
                    let bsdf = BSDFClosure {
                        frame,
                        bsdf: opt_bsdf.unwrap(),
                    };
                    let wo = -ray.d;
                    sppm_pixel.vp = Some(VisiblePoint {
                        p,
                        beta: Spectrum::one(),
                        bsdf,
                        wo,
                    });
                }
            });
            {
                let mut bound = Bounds3f::default();
                let mut max_radius = 0.0 as Float;
                for pixel in &pixels {
                    if let Some(vp) = &pixel.vp {
                        let p_bound = Bounds3f {
                            min: vp.p - vec3(pixel.radius, pixel.radius, pixel.radius),
                            max: vp.p + vec3(pixel.radius, pixel.radius, pixel.radius),
                        };
                        bound.insert_box(&p_bound);
                        max_radius = max_radius.max(pixel.radius);
                    }
                }
                println!("{:?} {}", bound, max_radius);
                let diag = bound.diagonal();
                let max_diag = glm::comp_max(&diag) as f64;
                let base_grid_res = (max_diag / max_radius) as u32;
                let mut grid_res = [0; 3];
                for i in 0..3 {
                    grid_res[i] = ((base_grid_res as f64 * diag[i] / max_diag) as u32).max(1);
                }
                grid = Some(VisiblePointGrid::new(&bound, grid_res, npixels));
                parallel_for(npixels, 256, |id| {
                    let sppm_pixel = unsafe { p_pixels.offset(id as isize).as_ref().unwrap() };
                    // let p = sppm_pixel.vp.as_ref().unwrap().p;
                    grid.as_ref().unwrap().insert(sppm_pixel);
                });
            }
            parallel_for(self.n_photons, 256, |id| {
                let sampler = unsafe {
                    p_photon_samplers
                        .offset(id as isize)
                        .as_mut()
                        .unwrap()
                        .as_mut()
                };
                let light = &scene.lights[0];
                let sample = light.sample_le(&[sampler.next2d(), sampler.next2d()]);
                let mut depth = 0;
                let mut ray = sample.ray;
                let mut beta = sample.le / (sample.pdf_dir * sample.pdf_pos)
                    * glm::dot(&sample.n, &ray.d).abs();
                loop {
                    if let Some(isct) = scene.shape.intersect(&ray) {
                        let ng = isct.ng;
                        let frame = Frame::from_normal(&ng);
                        let shape = isct.shape;
                        let opt_bsdf = shape.bsdf();
                        if opt_bsdf.is_none() {
                            break;
                        }
                        let p = ray.at(isct.t);
                        let bsdf = BSDFClosure {
                            frame,
                            bsdf: opt_bsdf.unwrap(),
                        };
                        let wo = -ray.d;
                        // println!("{} {} {}", p, depth, self.max_depth);
                        if depth >= self.max_depth {
                            break;
                        }
                        depth += 1;
                        {
                            // splat to grid
                            let grid = grid.as_ref().unwrap();
                            let h = grid.hash(&grid.to_grid(p));
                            let mut ap = grid.grid[h].0.load(Ordering::Relaxed);
                            while !ap.is_null() {
                                let node = unsafe { &*ap };
                                let pixel = node.pixel;
                                let wi = -ray.d;
                                let vp = pixel.vp.as_ref().unwrap();
                                let dist2 = {
                                    let v = vp.p - p;
                                    glm::dot(&v, &v)
                                };
                                if dist2 <= pixel.radius * pixel.radius {
                                    let phi = beta * vp.bsdf.evaluate(&vp.wo, &wi);
                                    for i in 0..Spectrum::N_SAMPLES {
                                        pixel.phi[i].fetch_add(phi[i] as f32, Ordering::SeqCst);
                                    }
                                    pixel.M.fetch_add(1, Ordering::SeqCst);
                                }
                                ap = node.next.load(Ordering::Relaxed);
                            }
                        }

                        if let Some(bsdf_sample) = bsdf.sample(&sampler.next2d(), &wo) {
                            let wi = &bsdf_sample.wi;
                            ray = Ray::spawn(&p, wi).offset_along_normal(&ng);
                            beta *= bsdf_sample.f * glm::dot(wi, &ng).abs() / bsdf_sample.pdf;
                        } else {
                            break;
                        }
                    } else {
                        break;
                    }
                }
            });
            parallel_for(npixels, 256, |id| {
                let p = unsafe { p_pixels.offset(id as isize).as_mut().unwrap() };
                let gamma = 2.0 / 3.0;
                #[allow(non_snake_case)]
                if p.M.load(Ordering::Relaxed) > 0 {
                    let N_new = p.N + (gamma * p.M.load(Ordering::Relaxed) as Float);
                    let R_new =
                        p.radius * (N_new / (p.N + p.M.load(Ordering::Relaxed) as Float)).sqrt();
                    let mut phi = Spectrum::zero();
                    for i in 0..Spectrum::N_SAMPLES {
                        phi[i] = p.phi[i].load(Ordering::Relaxed) as Float;
                        p.phi[i].store(0.0, Ordering::SeqCst);
                    }
                    p.tau = (p.tau + p.vp.as_ref().unwrap().beta * phi) * (R_new * R_new)
                        / (p.radius * p.radius);
                    p.N = N_new;
                    p.radius = R_new;
                }
            });
        }
        parallel_for(npixels, 256, |id| {
            let x = (id as u32) % scene.camera.resolution().x;
            let y = (id as u32) / scene.camera.resolution().x;
            let pixel = uvec2(x, y);
            let p = unsafe { p_pixels.offset(id as isize).as_ref().unwrap() };
            let l =
                p.tau / ((self.iterations * self.n_photons) as Float * PI * p.radius * p.radius);

            film.add_sample(&pixel, &l, 1.0);
        });
        film
    }
}

mod bdpt {
    use super::*;

    #[derive(Clone, Copy)]
    pub struct VertexBase {
        pdf_fwd: Float,
        pdf_rev: Float,
        delta: bool,
        beta: Spectrum,
        wo: Vec3,
        p: Vec3,
        n: Vec3,
    }
    #[derive(Copy, Clone)]
    pub struct SurfaceVertex<'a> {
        bsdf: BSDFClosure<'a>,
        n: Vec3,
        base: VertexBase,
    }
    #[derive(Copy, Clone)]
    pub struct CameraVertex<'a> {
        camera: &'a dyn Camera,
        base: VertexBase,
    }
    #[derive(Copy, Clone)]
    pub struct LightVertex<'a> {
        light: &'a dyn Light,
        base: VertexBase,
    }
    #[derive(Copy, Clone)]
    pub enum Vertex<'a> {
        Camera(CameraVertex<'a>),
        Light(LightVertex<'a>),
        Surface(SurfaceVertex<'a>),
    }

    impl<'a> Vertex<'a> {
        fn create_camera_vertex(
            camera: &'a dyn Camera,
            ray: &Ray,
            beta: Spectrum,
            pdf_fwd: Float,
        ) -> Self {
            Self::Camera(CameraVertex {
                camera,
                base: VertexBase {
                    wo: glm::zero(),
                    pdf_fwd,
                    pdf_rev: 0.0,
                    delta: false,
                    beta,
                    p: ray.o,
                    n: camera.n(), // ?????
                },
            })
        }
        fn create_light_vertex(
            light: &'a dyn Light,
            p: Vec3,
            beta: Spectrum,
            pdf_fwd: Float,
        ) -> Self {
            Self::Light(LightVertex {
                light,
                base: VertexBase {
                    wo: glm::zero(),
                    pdf_fwd,
                    pdf_rev: 0.0,
                    delta: false,
                    beta,
                    p,
                    n: vec3(0.0, 0.0, 0.0), // ?????
                },
            })
        }
        fn create_surface_vertex(
            beta: Spectrum,
            p: Vec3,
            bsdf: BSDFClosure<'a>,
            wo: Vec3,
            n: Vec3,
            mut pdf_fwd: Float,
            prev: &Vertex<'a>,
        ) -> Self {
            let mut v = Self::Surface(SurfaceVertex {
                bsdf,
                n,
                base: VertexBase {
                    beta,
                    wo,
                    pdf_fwd: 0.0,
                    pdf_rev: 0.0,
                    delta: false,
                    p,
                    n,
                },
            });
            pdf_fwd = prev.convert_pdf_to_area(pdf_fwd, &v);
            v.base_mut().pdf_fwd = pdf_fwd;
            v
        }
        pub fn is_delta_light(&self) -> bool {
            match self {
                Self::Light(v) => (v.light.flags() | LightFlags::DELTA) != LightFlags::NONE,
                _ => unreachable!(),
            }
        }
        pub fn on_surface(&self) -> bool {
            match self {
                Self::Camera(_v) => false,
                Self::Surface(_v) => true,
                Self::Light(_v) => false, //????
            }
        }
        pub fn as_camera(&self) -> Option<&CameraVertex> {
            match self {
                Self::Camera(v) => Some(v),
                _ => None,
            }
        }
        pub fn as_surface(&self) -> Option<&SurfaceVertex> {
            match self {
                Self::Surface(v) => Some(v),
                _ => None,
            }
        }
        pub fn as_light(&self) -> Option<&LightVertex> {
            match self {
                Self::Light(v) => Some(v),
                _ => None,
            }
        }
        pub fn base(&self) -> &VertexBase {
            match self {
                Self::Surface(v) => &v.base,
                Self::Light(v) => &v.base,
                Self::Camera(v) => &v.base,
            }
        }
        pub fn base_mut(&mut self) -> &mut VertexBase {
            match self {
                Self::Surface(v) => &mut v.base,
                Self::Light(v) => &mut v.base,
                Self::Camera(v) => &mut v.base,
            }
        }
        pub fn pdf_light_origin(&self, scene: &Scene, next: &Vertex) -> Float {
            match self {
                Vertex::Light(v) => {
                    let light_pdf = scene.light_distr.pdf(v.light);
                    let wi = glm::normalize(&(next.p() - self.p()));
                    let (pdf_pos, _) = v.light.pdf_li(
                        &wi,
                        &ReferencePoint {
                            p: self.p(),
                            n: self.n(),
                        },
                    );
                    light_pdf * pdf_pos
                }
                _ => unreachable!(),
            }
        }
        pub fn pdf_light(&self, scene: &Scene, next: &Vertex) -> Float {
            match self {
                Vertex::Light(v) => {
                    let ray = Ray::spawn_to(&self.p(), &next.p());
                    let (_pdf_pos, pdf_dir) = v.light.pdf_le(&ray);
                    self.convert_pdf_to_area(pdf_dir, next)
                }
                _ => unreachable!(),
            }
        }
        pub fn pdf(&self, scene: &Scene, prev: Option<&Vertex<'a>>, next: &Vertex<'a>) -> Float {
            let p2 = next.p();
            let p = self.p();
            let pdf = match self {
                Vertex::Surface(v) => {
                    let p1 = prev.unwrap().p();
                    let wo = glm::normalize(&(p1 - p));
                    let wi = glm::normalize(&(p2 - p));
                    v.bsdf.evaluate_pdf(&wo, &wi)
                }
                Vertex::Light(_) => self.pdf_light(scene, next),
                _ => unreachable!(),
            };
            self.convert_pdf_to_area(pdf, next)
        }
        pub fn f(&self, next: &Vertex<'a>, _mode: TransportMode) -> Spectrum {
            let v1 = self.as_surface().unwrap();
            // let v2 = next.as_surface().unwrap();
            let wi = glm::normalize(&(next.p() - self.p()));
            v1.bsdf.evaluate(&self.base().wo, &wi)
        }
        pub fn beta(&self) -> Spectrum {
            self.base().beta
        }
        pub fn pdf_fwd(&self) -> Float {
            self.base().pdf_fwd
        }
        pub fn p(&self) -> Vec3 {
            self.base().p
        }
        pub fn n(&self) -> Vec3 {
            self.base().n
        }
        pub fn le(&self, scene: &'a Scene, prev: &Vertex<'a>) -> Spectrum {
            if let Some(v) = prev.as_light() {
                v.light.le(&Ray::spawn_to(&self.p(), &prev.p()))
            } else {
                Spectrum::zero()
            }
        }
        pub fn convert_pdf_to_area(&self, mut pdf: Float, v2: &Vertex) -> Float {
            let w = v2.p() - self.p();
            let inv_dist2 = 1.0 / glm::dot(&w, &w);
            if v2.on_surface() {
                pdf *= glm::dot(&v2.n(), &(w * inv_dist2.sqrt())).abs();
            }
            pdf * inv_dist2
        }
        pub fn connectible(&self) -> bool {
            match self {
                Vertex::Light(v) => {
                    let flags = v.light.flags();
                    (flags | LightFlags::DELTA_DIRECTION) == LightFlags::NONE
                }
                Vertex::Camera(_) => true,
                Vertex::Surface(v) => true,
            }
        }
    }
    pub enum TransportMode {
        IMPORTANCE,
        RADIANCE,
    }
    pub fn random_walk<'a>(
        scene: &'a Scene,
        mut ray: Ray,
        sampler: &mut dyn Sampler,
        mut beta: Spectrum,
        pdf: Float,
        max_depth: usize,
        _mode: TransportMode,
        path: &mut Path<'a>,
    ) {
        assert!(pdf > 0.0);
        assert!(path.len() == 1);
        if max_depth == 0 {
            return;
        }
        let mut pdf_fwd = pdf;
        let mut pdf_rev = 0.0;
        let mut depth = 0usize;
        loop {
            if let Some(isct) = scene.shape.intersect(&ray) {
                let ng = isct.ng;
                let frame = Frame::from_normal(&ng);
                let shape = isct.shape;
                let opt_bsdf = shape.bsdf();
                if opt_bsdf.is_none() {
                    break;
                }
                let p = ray.at(isct.t);
                let bsdf = BSDFClosure {
                    frame,
                    bsdf: opt_bsdf.unwrap(),
                };
                let wo = -ray.d;
                if depth >= max_depth {
                    break;
                }
                let prev_index = depth;
                // let this_index = prev_index + 1;
                let prev = &mut path[prev_index];
                let vertex =
                    Vertex::create_surface_vertex(beta, p, bsdf.clone(), wo, ng, pdf_fwd, prev);
                // pdf_rev = vertex.pdf(scene, prev, next)
                depth += 1;

                if let Some(bsdf_sample) = bsdf.sample(&sampler.next2d(), &wo) {
                    pdf_fwd = bsdf_sample.pdf;
                    let wi = &bsdf_sample.wi;
                    {
                        pdf_rev = bsdf.evaluate_pdf(&wi, &wo);
                        prev.base_mut().pdf_rev = vertex.convert_pdf_to_area(pdf_rev, prev);
                    }
                    ray = Ray::spawn(&p, wi).offset_along_normal(&ng);
                    beta *= bsdf_sample.f * glm::dot(wi, &ng).abs() / bsdf_sample.pdf;
                    path.push(vertex);
                } else {
                    break;
                }
            } else {
                break;
            }
        }
    }
    pub fn generate_camera_path<'a>(
        scene: &'a Scene,
        pixel: &glm::UVec2,
        sampler: &mut dyn Sampler,
        max_depth: usize,
        path: &mut Path<'a>,
    ) {
        path.clear();
        let camera = scene.camera.as_ref();
        let (ray, beta) = camera.generate_ray(pixel, sampler);
        let vertex = Vertex::create_camera_vertex(camera, &ray, beta, 1.0);
        path.push(vertex);
        let (_pdf_pos, pdf_dir) = camera.pdf_we(&ray);
        random_walk(
            scene,
            ray,
            sampler,
            beta,
            pdf_dir,
            max_depth - 1,
            TransportMode::RADIANCE,
            path,
        );
    }
    pub fn generate_light_path<'a>(
        scene: &'a Scene,
        sampler: &mut dyn Sampler,
        max_depth: usize,
        path: &mut Path<'a>,
    ) {
        path.clear();
        let (light, light_pdf) = scene.light_distr.sample(sampler.next1d());
        let sample = light.sample_le(&[sampler.next2d(), sampler.next2d()]);
        let le = sample.le;
        let beta = le / (sample.pdf_dir * sample.pdf_pos * light_pdf)
            * glm::dot(&sample.ray.d, &sample.n).abs();
        let vertex =
            Vertex::create_light_vertex(light, sample.ray.o, le, light_pdf * sample.pdf_pos);
        path.push(vertex);
        random_walk(
            scene,
            sample.ray,
            sampler,
            beta,
            sample.pdf_dir,
            max_depth - 1,
            TransportMode::IMPORTANCE,
            path,
        );
    }
    pub type Path<'a> = Vec<Vertex<'a>>;
    pub fn geometry_term(scene: &Scene, v1: &Vertex, v2: &Vertex) -> Float {
        let mut wi = v1.p() - v2.p();
        let dist2: Float = glm::dot(&wi, &wi);
        wi /= dist2.sqrt();
        let ray = Ray::spawn_to(&v1.p(), &v2.p()).offset_along_normal(&v1.n());
        if scene.shape.occlude(&ray) {
            0.0
        } else {
            (glm::dot(&wi, &v1.n()) * glm::dot(&wi, &v2.n()) / dist2).abs()
        }
    }
    #[derive(Debug, Clone, Copy)]
    pub struct ConnectionStrategy {
        pub s: usize,
        pub t: usize,
    }
    pub struct Scratch<'a> {
        new_light_path: Path<'a>,
        new_eye_path: Path<'a>,
        // strat:Option<ConnectionStrategy>,
    }
    impl<'a> Scratch<'a> {
        pub fn new() -> Self {
            Self {
                new_light_path: Vec::new(),
                new_eye_path: Vec::new(),
            }
        }
    }
    pub fn mis_weight<'a>(
        scene: &'a Scene,
        strat: ConnectionStrategy,
        original_light_path: &Path<'a>,
        original_eye_path: &Path<'a>,
        sampled: Option<Vertex<'a>>,
        scratch: &mut Scratch<'a>,
    ) -> Float {
        let s = strat.s;
        let t = strat.t;
        // 1.0 / (s + t - 1) as Float
        if s + t == 2 {
            return 1.0;
        }
        let eye_path = &mut scratch.new_eye_path;
        let light_path = &mut scratch.new_light_path;
        eye_path.clear();
        light_path.clear();
        for i in 0..s {
            light_path.push(original_light_path[i]);
        }
        for i in 0..t {
            eye_path.push(original_eye_path[i]);
            // println!(
            //     "{} {}",
            //     original_eye_path[i].base().pdf_fwd,
            //     original_eye_path[i].base().pdf_rev
            // );
        }
        if s == 1 {
            light_path.push(sampled.unwrap());
        } else if t == 1 {
            eye_path.push(sampled.unwrap());
        }
        // update vertices
        {
            let qs = if s > 0 {
                &mut light_path[s - 1] as *mut Vertex<'a>
            } else {
                std::ptr::null_mut()
            };
            let qs_minus = if s > 1 {
                &mut light_path[s - 2] as *mut Vertex<'a>
            } else {
                std::ptr::null_mut()
            };
            let pt = if t > 0 {
                &mut eye_path[t - 1] as *mut Vertex<'a>
            } else {
                std::ptr::null_mut()
            };
            let pt_minus = if t > 1 {
                &mut eye_path[t - 2] as *mut Vertex<'a>
            } else {
                std::ptr::null_mut()
            };
            // p0....pt-1 pt  qs qs-1 ...q0
            if !pt.is_null() {
                let pt = unsafe { pt.as_mut().unwrap() };
                let qs_minus = unsafe { qs_minus.as_ref() };
                let qs = unsafe { qs.as_ref() };
                let pt_minus = unsafe { &*pt_minus };
                pt.base_mut().delta = false;
                // pt-1 pt<- qs qs-1
                pt.base_mut().pdf_rev = if s > 0 {
                    qs.unwrap().pdf(scene, qs_minus, pt)
                } else {
                    pt.pdf_light_origin(scene, pt_minus)
                };
            }

            if !pt_minus.is_null() {
                let pt = unsafe { pt.as_mut().unwrap() };
                let qs = unsafe { qs.as_ref() };
                let pt_minus = unsafe { pt_minus.as_ref().unwrap() };
                // pt-1 <- pt qs qs-1
                pt.base_mut().pdf_rev = if s > 0 {
                    pt.pdf(scene, qs, pt_minus)
                } else {
                    // pt-1 <- pt
                    pt.pdf_light(scene, pt_minus)
                };
            }

            if !qs.is_null() {
                let qs = unsafe { qs.as_mut().unwrap() };
                let pt = unsafe { pt.as_mut().unwrap() };
                let pt_minus = unsafe { pt_minus.as_ref() };
                qs.base_mut().delta = false;
                // pt-1 pt-> qs qs-1
                qs.base_mut().pdf_rev = pt.pdf(scene, pt_minus, qs);
            }
            if !qs_minus.is_null() {
                let qs = unsafe { qs.as_mut().unwrap() };
                let pt = unsafe { pt.as_ref() };
                let qs_minus = unsafe { qs_minus.as_mut().unwrap() };
                qs_minus.base_mut().pdf_rev = qs.pdf(scene, pt, qs_minus);
            }
        }

        let mut sum_ri = 0.0;
        let remap = |x| {
            if x == 0.0 {
                1.0
            } else {
                x
            }
        };
        {
            // camera path
            let mut ri = 1.0;
            for i in (2..=t - 1).rev() {
                ri *= remap(eye_path[i].base().pdf_rev) / remap(eye_path[i].base().pdf_fwd);
                if !eye_path[i].base().delta {
                    sum_ri += ri;
                }
            }
        }
        {
            let mut ri = 1.0;
            for i in (0..=s - 1).rev() {
                ri *= remap(light_path[i].base().pdf_rev) / remap(light_path[i].base().pdf_fwd);
                let delta_light = if i > 0 {
                    light_path[i - 1].base().delta
                } else {
                    light_path[0].is_delta_light()
                };
                if !light_path[i].base().delta && !delta_light {
                    sum_ri += ri;
                }
            }
        }
        // println!("{}", 1.0 / (1.0 + sum_ri));
        1.0 / (1.0 + sum_ri)
    }
    pub fn connect_paths<'a>(
        scene: &'a Scene,
        strat: ConnectionStrategy,
        light_path: &Path<'a>,
        eye_path: &Path<'a>,
        sampler: &mut dyn Sampler,
        scratch: &mut Scratch<'a>,
    ) -> Spectrum {
        let s = strat.s;
        let t = strat.t;
        let mut sampled: Option<Vertex> = None;
        let mut l = Spectrum::zero();
        if t > 1 && s != 0 && eye_path[t - 1].as_light().is_some() {
            return Spectrum::zero();
        } else if s == 0 {
            let pt = &eye_path[t - 1];
            l = pt.beta() * pt.le(scene, &eye_path[t - 2]);
        } else if t == 1 {
            unreachable!();
        } else if s == 1 {
            let pt = &eye_path[t - 1];
            if pt.connectible() {
                let (light, light_pdf) = scene.light_distr.sample(sampler.next1d());
                let p_ref = ReferencePoint {
                    p: pt.p(),
                    n: pt.n(),
                };
                let light_sample = light.sample_li(&sampler.next2d(), &p_ref);
                if !light_sample.li.is_black() {
                    {
                        let mut v = Vertex::create_light_vertex(
                            light,
                            light_sample.p,
                            light_sample.li / (light_sample.pdf * light_pdf),
                            0.0,
                        );
                        v.base_mut().pdf_fwd = v.pdf_light_origin(scene, pt);
                        sampled = Some(v);
                    }
                    {
                        let sampled = sampled.as_ref().unwrap();
                        l = pt.beta() * pt.f(sampled, TransportMode::RADIANCE) * sampled.beta();
                    }

                    if pt.on_surface() {
                        l *= glm::dot(&light_sample.wi, &p_ref.n).abs();
                    }
                    if scene.shape.occlude(&light_sample.shadow_ray) {
                        l *= 0.0;
                    }
                    // li += beta
                    //     * bsdf.evaluate(&wo, &light_sample.wi)
                    //     * glm::dot(&ng, &light_sample.wi).abs()
                    //     * light_sample.li
                    //     / light_sample.pdf;
                }
            }
        } else {
            let pt = &eye_path[t - 1];
            let qs = &light_path[s - 1];
            if pt.connectible() && qs.connectible() {
                l = qs.beta()
                    * pt.beta()
                    * pt.f(qs, TransportMode::RADIANCE)
                    * qs.f(pt, TransportMode::IMPORTANCE);
                if !l.is_black() {
                    l *= geometry_term(scene, pt, qs);
                }
            }
        }
        let mis_weight = if l.is_black() {
            0.0
        } else {
            bdpt::mis_weight(scene, strat, light_path, eye_path, sampled, scratch)
        };
        l * mis_weight
    }
}
struct BDPT {
    spp: usize,
    max_depth: usize,
    debug: bool,
}

impl Integrator for BDPT {
    type Output = Film;
    fn render(&mut self, scene: &Scene) -> Self::Output {
        let npixels = (scene.camera.resolution().x * scene.camera.resolution().y) as usize;
        let film = Film::new(&scene.camera.resolution());
        let mut pyramid = Vec::new();
        for _t in 2..=self.max_depth + 2 {
            for _s in 0..self.max_depth + 2 {
                pyramid.push(Film::new(&scene.camera.resolution()));
            }
        }
        let get_index = |s, t| (t - 2) as usize * (3 + self.max_depth) + s as usize;
        parallel_for(npixels, 256, |id| {
            let mut sampler = PCGSampler { rng: PCG::new(id) };
            let x = (id as u32) % scene.camera.resolution().x;
            let y = (id as u32) / scene.camera.resolution().x;
            let pixel = uvec2(x, y);
            let mut acc_li = Spectrum::zero();
            let mut camera_path = vec![];
            let mut light_path = vec![];
            let mut debug_acc = vec![];
            if self.debug {
                for _t in 2..=self.max_depth + 2 {
                    for _s in 0..=self.max_depth + 2 {
                        debug_acc.push(Spectrum::zero());
                    }
                }
            }
            let mut scratch = bdpt::Scratch::new();
            for _ in 0..self.spp {
                bdpt::generate_camera_path(
                    scene,
                    &pixel,
                    &mut sampler,
                    self.max_depth + 2,
                    &mut camera_path,
                );
                bdpt::generate_light_path(scene, &mut sampler, self.max_depth + 1, &mut light_path);
                for t in 2..=camera_path.len() as isize {
                    for s in 0..=light_path.len() as isize {
                        let depth = s + t - 2;
                        if (s == 1 && t == 1) || depth < 0 || depth > self.max_depth as isize {
                            continue;
                        }
                        let li = bdpt::connect_paths(
                            scene,
                            bdpt::ConnectionStrategy {
                                s: s as usize,
                                t: t as usize,
                            },
                            &mut light_path,
                            &mut camera_path,
                            &mut sampler,
                            &mut scratch,
                        );
                        if self.debug {
                            debug_acc[get_index(s, t)] += li;
                        }
                        acc_li += li;
                    }
                }
            }
            acc_li = acc_li / (self.spp as Float);

            film.add_sample(&uvec2(x, y), &acc_li, 1.0);

            if self.debug {
                for t in 2..=(self.max_depth + 2) as isize {
                    for s in 0..=(self.max_depth + 2) as isize {
                        let depth = s + t - 2;
                        if (s == 1 && t == 1) || depth < 0 || depth > self.max_depth as isize {
                            continue;
                        }
                        let idx = get_index(s, t);
                        pyramid[idx].add_sample(
                            &uvec2(x, y),
                            &(debug_acc[idx] / (self.spp as Float) as Float),
                            1.0,
                        );
                    }
                }
            }
        });
        if self.debug {
            for t in 2..=(self.max_depth + 2) as isize {
                for s in 0..=(self.max_depth + 2) as isize {
                    let depth = s + t - 2;
                    if (s == 1 && t == 1) || depth < 0 || depth > self.max_depth as isize {
                        continue;
                    }
                    let idx = get_index(s, t);
                    let film = &pyramid[idx];
                    let img = film.to_rgb_image();
                    img.save(format!("bdpt-d{}-s{}-t{}.png", depth, s, t))
                        .unwrap();
                }
            }
        }
        film
    }
}
#[macro_use]
mod nn {
    use core::f64;
    use std::marker::PhantomData;

    use super::*;
    // unsafe fn atomic_fetch_add_f32(p: *mut f32, val: f32) -> f32 {
    //     use std::mem::size_of;
    //     let at = &*std::mem::transmute::<*mut f32, *mut AtomicFloat>(p);
    //     assert!(size_of::<AtomicFloat>() == size_of::<f32>());
    //     at.fetch_add(val, Ordering::SeqCst)
    // }
    #[derive(Clone, Copy)]
    pub struct Dual<T> {
        val: T,
        deriv: T,
    }
    pub trait ActivationFunction {
        fn forward(&self, x: f32) -> f32;
        fn backward(&self, out: Dual<f32>, x: f32) -> f32;
        fn new() -> Self;
    }
    pub struct Sigmoid {}
    impl ActivationFunction for Sigmoid {
        fn forward(&self, x: f32) -> f32 {
            1.0 / (1.0 + (-x).exp())
        }
        fn backward(&self, out: Dual<f32>, _x: f32) -> f32 {
            (1.0 - out.val) * out.val * out.deriv
        }
        fn new() -> Self {
            Self {}
        }
    }
    pub struct Relu {}
    impl ActivationFunction for Relu {
        fn forward(&self, x: f32) -> f32 {
            x.max(0.0)
        }
        fn backward(&self, out: Dual<f32>, x: f32) -> f32 {
            if x >= 0.0 {
                out.deriv
            } else {
                0.0
            }
        }
        fn new() -> Self {
            Self {}
        }
    }
    pub struct Tanh {}
    impl ActivationFunction for Tanh {
        fn forward(&self, x: f32) -> f32 {
            x.tanh()
        }
        fn backward(&self, out: Dual<f32>, _x: f32) -> f32 {
            (1.0 - out.val * out.val) * out.deriv
        }
        fn new() -> Self {
            Self {}
        }
    }
    pub struct Linear {}
    impl ActivationFunction for Linear {
        fn forward(&self, x: f32) -> f32 {
            x
        }
        fn backward(&self, out: Dual<f32>, x: f32) -> f32 {
            out.deriv
        }
        fn new() -> Self {
            Self {}
        }
    }
    pub trait VectorizedOptimizer: Clone {
        fn step(&mut self, val: &mut [f32], grad: &[f32]);
    }
    pub trait OptimizerImpl: Default + Clone {
        fn step(&mut self, val: &mut f32, grad: f32);
        // type Vectorized;
    }
    #[derive(Clone)]
    pub struct Optimizer<T, Impl: OptimizerImpl> {
        // fn step(&mut self, val:&mut T, grad: &T);
        data: PhantomData<T>,
        impl_: Vec<Impl>,
    }
    impl<Impl: OptimizerImpl, const M: usize, const N: usize> Optimizer<na::SMatrix<f32, M, N>, Impl> {
        fn new(opt: Impl) -> Self {
            Self {
                data: PhantomData {},
                impl_: vec![opt; M * N],
            }
        }
        fn step(&mut self, val: &mut na::SMatrix<f32, M, N>, grad: &na::SMatrix<f32, M, N>) {
            assert!(self.impl_.len() == M * N);
            // if self.impl_.len() != M * N {
            //     self.impl_.resize(M * N, self.impl_[0].clone());
            // }
            for i in 0..(M * N) {
                self.impl_[i].step(&mut (*val)[i], grad[i]);
            }
        }
    }
    #[derive(Copy, Clone)]
    pub struct SGD {
        pub learning_rate: f32,
    }
    impl Default for SGD {
        fn default() -> Self {
            SGD {
                learning_rate: 0.01,
            }
        }
    }
    impl OptimizerImpl for SGD {
        fn step(&mut self, val: &mut f32, grad: f32) {
            *val -= self.learning_rate * grad.max(-100.0).min(100.0);
        }
    }
    #[derive(Copy, Clone)]
    pub struct Momentum {
        pub a: f32,
        pub v: f32,
        pub learning_rate: f32,
    }
    impl Default for Momentum {
        fn default() -> Self {
            Momentum {
                learning_rate: 0.01,
                v: 0.0,
                a: 0.9,
            }
        }
    }
    impl OptimizerImpl for Momentum {
        fn step(&mut self, val: &mut f32, grad: f32) {
            self.v = self.a * self.v + self.learning_rate * grad.min(100.0).max(-100.0);
            *val -= self.v;
        }
    }
    #[derive(Copy, Clone)]
    pub struct Batch<O: OptimizerImpl> {
        pub opt: O,
        pub batch_size: u32,
        pub count: u32,
        pub sum: f32,
    }
    impl<O: OptimizerImpl> Default for Batch<O> {
        fn default() -> Self {
            Batch {
                opt: Default::default(),
                batch_size: 1,
                count: 0,
                sum: 0.0,
            }
        }
    }
    impl<O: OptimizerImpl> OptimizerImpl for Batch<O> {
        fn step(&mut self, val: &mut f32, grad: f32) {
            // self.v = self.a * self.v + self.learning_rate * grad;
            // *val -= self.v;
            self.sum += grad;
            self.count += 1;
            if self.count >= self.batch_size {
                let grad = self.sum / self.count as f32;
                self.opt.step(val, grad);
                self.sum = 0.0;
                self.count = 0;
            }
        }
    }
    #[derive(Copy, Clone)]
    pub struct Adam {
        pub learning_rate: f32,
        pub m: f64,
        pub v: f64,
        pub beta1: f64,
        pub beta2: f64,
        pub t: i32,
    }
    impl Default for Adam {
        fn default() -> Self {
            Adam {
                learning_rate: 0.01,
                m: 0.0,
                v: 0.0,
                beta1: 0.9,
                beta2: 0.999,
                t: 0,
            }
        }
    }
    impl OptimizerImpl for Adam {
        fn step(&mut self, val: &mut f32, grad: f32) {
            let grad = grad as f64;
            self.t += 1;
            // *val -= self.learning_rate * grad;
            self.m = self.beta1 * self.m + (1.0 - self.beta1) * grad;
            self.v = self.beta2 * self.v + (1.0 - self.beta2) * grad * grad;
            let m_tilde = self.m / (1.0 - self.beta1.powi(self.t));
            let v_tilde = self.v / (1.0 - self.beta2.powi(self.t));

            *val -= self.learning_rate * (m_tilde / (v_tilde.sqrt() + 1e-8f64)) as f32;
        }
    }
    #[macro_use]
    pub mod mlp {
        use super::*;
        // pub struct
        // pub struct MLPEvalContext<const I: usize, const O:usize> {
        //     x:na::DMatrix<f32>,
        //     scratch:na::DMatrix<f32>,

        // }
        pub trait Module<const I: usize, const O: usize> {
            type Temp;
            fn forward(
                &self,
                x: &na::SVector<f32, I>,
                temp: Option<&mut Self::Temp>,
            ) -> na::SVector<f32, O>;
            fn backward(
                &mut self,
                x: &na::SVector<f32, I>,
                out: Dual<&na::SVector<f32, O>>,
                temp: &Self::Temp,
            ) -> na::SVector<f32, I>;
            fn create_temp(&self) -> Self::Temp;
        }
        #[derive(Clone)]
        pub struct Activation<F: ActivationFunction> {
            pub f: F,
        }
        impl<F: ActivationFunction, const N: usize> Module<N, N> for Activation<F> {
            type Temp = ();
            fn create_temp(&self) -> Self::Temp {}
            fn forward(
                &self,
                x: &na::SVector<f32, N>,
                _temp: Option<&mut Self::Temp>,
            ) -> na::SVector<f32, N> {
                x.map(|t| self.f.forward(t))
            }
            fn backward(
                &mut self,
                x: &na::SVector<f32, N>,
                out: Dual<&na::SVector<f32, N>>,
                _temp: &Self::Temp,
            ) -> na::SVector<f32, N> {
                let mut grad: na::SVector<f32, N> = na::zero();
                for i in 0..N {
                    grad[i] = self.f.backward(
                        Dual {
                            val: out.val[i],
                            deriv: out.deriv[i],
                        },
                        x[i],
                    );
                }
                // println!("{}", grad);
                grad
            }
        }
        #[derive(Clone)]
        pub struct Linear<Opt: OptimizerImpl, const I: usize, const O: usize> {
            pub weights: na::SMatrix<f32, O, I>,
            pub bias: na::SVector<f32, O>,
            pub weights_opt: Optimizer<na::SMatrix<f32, O, I>, Opt>,
            pub bias_opt: Optimizer<na::SVector<f32, O>, Opt>,
        }
        impl<Opt, const I: usize, const O: usize> Linear<Opt, I, O>
        where
            Opt: OptimizerImpl,
        {
            pub fn new(opt: Opt) -> Self {
                use rand::distributions::Distribution;
                use rand::Rng;
                use statrs::distribution::Normal;
                let mut rng = rand::thread_rng();
                let n = Normal::new(0.0, (2.0 / (I + O) as f64).sqrt()).unwrap();
                let mut weights: na::SMatrix<f32, O, I> = na::zero();
                let bias = na::zero();
                weights = weights.map(|_| n.sample(&mut rng) as f32);
                Self {
                    weights,
                    bias,
                    weights_opt: Optimizer::new(opt.clone()),
                    bias_opt: Optimizer::new(opt.clone()),
                }
            }
        }
        impl<Opt, const I: usize, const O: usize> Module<I, O> for Linear<Opt, I, O>
        where
            Opt: OptimizerImpl,
        {
            type Temp = (); //na::SVector<f32, O>;
            fn create_temp(&self) -> Self::Temp {}
            fn forward(
                &self,
                x: &na::SVector<f32, I>,
                _temp: Option<&mut Self::Temp>,
            ) -> na::SVector<f32, O> {
                self.weights * x + self.bias
            }
            fn backward(
                &mut self,
                x: &na::SVector<f32, I>,
                out: Dual<&na::SVector<f32, O>>,
                _temp: &Self::Temp,
            ) -> na::SVector<f32, I> {
                let dbias = out.deriv;
                let dw = out.deriv * x.transpose();
                let dx = self.weights.transpose() * out.deriv;
                // let dx =(out.deriv.transpose() * self.weights).transpose();
                //  println!("{} {}", dw, dx);
                self.bias_opt.step(&mut self.bias, dbias);
                self.weights_opt.step(&mut self.weights, &dw);
                dx
            }
        }
        #[derive(Clone)]
        pub struct Dense<F: ActivationFunction, Opt: OptimizerImpl, const I: usize, const O: usize> {
            pub linear: Linear<Opt, I, O>,
            pub act: Activation<F>,
        }
        impl<F: ActivationFunction, Opt, const I: usize, const O: usize> Dense<F, Opt, I, O>
        where
            Opt: OptimizerImpl,
        {
            pub fn new(opt: Opt) -> Self {
                Self {
                    linear: Linear::new(opt.clone()),
                    act: Activation { f: F::new() },
                }
            }
        }
        impl<F: ActivationFunction, Opt, const I: usize, const O: usize> Module<I, O>
            for Dense<F, Opt, I, O>
        where
            Opt: OptimizerImpl,
        {
            type Temp = na::SVector<f32, O>;
            fn create_temp(&self) -> Self::Temp {
                na::zero()
            }
            fn forward(
                &self,
                x: &na::SVector<f32, I>,
                temp: Option<&mut Self::Temp>,
            ) -> na::SVector<f32, O> {
                let v = self.linear.forward(x, None);
                if let Some(temp) = temp {
                    *temp = v;
                }
                self.act.forward(&v, None)
            }
            fn backward(
                &mut self,
                x: &na::SVector<f32, I>,
                out: Dual<&na::SVector<f32, O>>,
                linear_out: &Self::Temp,
            ) -> na::SVector<f32, I> {
                let linear_grad = self.act.backward(linear_out, out, &());
                let dlinear = Dual {
                    val: linear_out,
                    deriv: &linear_grad,
                };
                self.linear.backward(x, dlinear, &())
            }
        }

        #[derive(Clone)]
        pub struct CombineModules<A, B, const I: usize, const H: usize, const O: usize> {
            pub a: A,
            pub b: B,
        }
        impl<A, B, const I: usize, const H: usize, const O: usize> Module<I, O>
            for CombineModules<A, B, I, H, O>
        where
            A: Module<I, H>,
            B: Module<H, O>,
        {
            type Temp = (
                <A as Module<I, H>>::Temp,
                <B as Module<H, O>>::Temp,
                na::SVector<f32, H>,
            );
            fn create_temp(&self) -> Self::Temp {
                (self.a.create_temp(), self.b.create_temp(), na::zero())
            }
            fn forward(
                &self,
                x: &na::SVector<f32, I>,
                temp: Option<&mut Self::Temp>,
            ) -> na::SVector<f32, O> {
                let p_temp = temp.map_or(std::ptr::null_mut(), |t| t as *mut Self::Temp);
                let v = unsafe {
                    let temp = p_temp.as_mut();
                    self.a.forward(x, temp.map(|y| &mut y.0))
                };
                if !p_temp.is_null() {
                    unsafe {
                        let temp = p_temp.as_mut();
                        temp.unwrap().2 = v;
                    }
                }
                unsafe {
                    let temp = p_temp.as_mut();
                    self.b.forward(&v, temp.map(|y| &mut y.1))
                }
            }
            fn backward(
                &mut self,
                x: &na::SVector<f32, I>,
                out: Dual<&na::SVector<f32, O>>,
                temp: &Self::Temp,
            ) -> na::SVector<f32, I> {
                let gradb = self.b.backward(&temp.2, out, &temp.1);
                let db = Dual {
                    val: &temp.2,
                    deriv: &gradb,
                };
                self.a.backward(x, db, &temp.0)
            }
        }
    }
    use mlp::Module;
    pub struct Model<M: Module<I, O>, const I: usize, const O: usize> {
        pub m: M,
    }
    impl<M: Module<I, O>, const I: usize, const O: usize> Model<M, I, O> {
        // pub fn new(m: M) -> Self {
        //     Self { m }
        // }
        pub fn infer(&self, x: &na::SVector<f32, I>) -> na::SVector<f32, O> {
            self.m.forward(x, None)
        }
        pub fn train(&mut self, x: &na::SVector<f32, I>, target: na::SVector<f32, O>) -> f32 {
            let mut temp = self.m.create_temp();
            let y = self.m.forward(x, Some(&mut temp));
            let (loss, dy) = {
                let d = y - target;
                (d.component_mul(&d).mean(), 2.0 * d)
            };

            self.m.backward(
                x,
                Dual {
                    val: &y,
                    deriv: &dy,
                },
                &temp,
            );
            loss
        }
    }
    #[macro_export]
    macro_rules! sequential {
       ($layer:expr) => {
           $layer
       };
       ($layer0:expr, $($rest:expr), *)=>{
           CombineModules{
               a:$layer0,
               b:sequential!($($rest), *)
           }
       };
   }
    macro_rules! create_mlp_helper {
        ($opt_v:expr, $act:ty, $opt:ty, $dim1:expr, $dim2:expr)=>{
            crate::nn::mlp::Linear::<$opt, $dim1, $dim2>::new($opt_v)
        };
        ($opt_v:expr,$act:ty, $opt:ty,  $dim1:expr,$dim2:expr, $($rest:expr), +)=>{

            crate::nn::mlp::CombineModules{
                   a: crate::nn::mlp::Dense::<$act, $opt, $dim1, $dim2>::new($opt_v),
                   b:create_mlp_helper!($opt_v, $act, $opt, $dim2, $($rest), *)
               }
            };
       }
    macro_rules! get_last {
       ($x:expr) => {
           $x
       };
       ($x:expr, $($xs:expr), *)=> {
           get_last!($($xs), *)
       };
   }
    macro_rules! mlp_type_helper {
    ( $act:ty, $opt:ty, $dim1:expr, $dim2:expr)=>{
        crate::nn::mlp::Linear<$opt, $dim1, $dim2>
    };
    ($act:ty, $opt:ty,  $dim1:expr,$dim2:expr, $($rest:expr), +)=>{
        crate::nn::mlp::CombineModules<
            crate::nn::mlp::Dense<$act, $opt, $dim1, $dim2>,
        mlp_type_helper!( $act, $opt, $dim2, $($rest), *),
        $dim1, $dim2, {get_last!($($rest), *)}>
    };
   }
    #[macro_export]
    macro_rules! create_mlp {
    // ($name:ident, $act:ty, $opt:ty, $dim1:expr, $dim2:expr)=>{
    //     type $name = crate::nn::Model<
    //         inner:mlp_type_helper!($act, $opt, $dim1, $dim2)
    //     >;
    //     // Model::new(create_mlp_helper!($opt_v, $act, $opt, $($rest), *))
    // };
    ($name:ident, $act:ty, $opt:ty,$dim1:expr, $($rest:expr), +)=>{
            pub type $name = crate::nn::Model<
                    mlp_type_helper!($act, $opt,$dim1, $($rest), +),
                    $dim1,
                    {get_last!( $($rest), +)}
                >;
            impl $name {
                pub fn new(o:$opt)->Self {
                    Self{m:create_mlp_helper!(o, $act, $opt, $dim1, $($rest), +)}
                }
            }

    }
   }

    #[macro_export]
    macro_rules! position_encoding_func {
        ($name:ident, $N:expr, $E:expr) => {
            fn $name(v: &na::SVector<f32, $N>) -> na::SVector<f32, { $N + $N * $E * 2 }> {
                let mut u: na::SVector<f32, { $N + $N * $E * 2 }> = na::zero();
                for i in 0..$N {
                    for j in 0..$E {
                        let feq = 2.0f32.powi(j as i32);
                        u[i * $E + j] = (v[i] * feq).sin();
                        u[i * $E + j + $N * $E] = (v[i] * feq).cos();
                    }
                    u[$N * $E * 2 + i] = v[i];
                }
                u
            }
        };
    }
}
mod nrc {
    use super::*;
    use nalgebra as na;
    use nalgebra_glm as glm;
    use nn::mlp::Module;
    use nn::*;
    use rand::distributions::Distribution;
    use rand::Rng;
    use statrs::distribution::Normal;

    const FEATURE_SIZE: usize = 6;
    const INPUT_SIZE: usize = 3 + 2 + Spectrum::N_SAMPLES + 3;
    // type FeatureMat = na::SMatrix<f32, { FEATURE_SIZE }, 5>;
    // type FeatureVec1 = na::SVector<f32, { FEATURE_SIZE }>;
    type InputVec = na::SVector<f32, { INPUT_SIZE }>;
    type FeatureVec = na::SVector<f32, { INPUT_SIZE * FEATURE_SIZE * 2 + INPUT_SIZE }>;

    create_mlp!(
        NRCModel,
        Relu,
        SGD,
        { INPUT_SIZE * FEATURE_SIZE * 2 + INPUT_SIZE },
        64,
        64,
        64,
        // 64,
        // 64,
        // 64,
        { Spectrum::N_SAMPLES }
    );

    position_encoding_func!(position_encoder, INPUT_SIZE, FEATURE_SIZE);
    struct RadianceCache {
        model: NRCModel,
        bound: Bounds3f,
    }
    #[derive(Clone, Copy)]
    struct QueryRecord {
        n: Vec3,
        info: BSDFInfo,
        x: Vec3,
        dir: Vec2,
    }
    impl RadianceCache {
        fn new(opt: SGD) -> Self {
            Self {
                model: NRCModel::new(opt),
                bound: Bounds3f {
                    min: vec3(1.0, 1.0, 1.0) * -20.0,
                    max: vec3(1.0, 1.0, 1.0) * 20.0,
                },
            }
        }
        fn get_input_vec(r: &QueryRecord) -> InputVec {
            InputVec::from_iterator(
                r.x.into_iter()
                    .chain(r.dir.into_iter())
                    .chain(r.n.into_iter())
                    .chain(r.info.albedo.samples.into_iter())
                    .map(|x| *x as f32),
            )
        }
        fn infer(&self, r: &QueryRecord) -> Option<Spectrum> {
            if !self.bound.contains(&r.x) {
                return None;
            }
            let v = Self::get_input_vec(r);
            let s: Spectrum = self.model.infer(&position_encoder(&v)).into();
            Some(s)
        }
        fn train(&mut self, r: &QueryRecord, target: &Spectrum) {
            if !self.bound.contains(&r.x) {
                return;
            }
            let v = Self::get_input_vec(r);
            let _loss = self
                .model
                .train(&position_encoder(&v), target.samples.cast::<f32>());
            // println!("{} {}",v, _loss);
        }
    }
    pub struct CachedPathTracer {
        pub spp: u32,
        pub training_samples: u32,
        pub max_depth: u32,
    }
    #[derive(Copy, Clone)]
    struct Vertex {
        x: Vec3,
        dir: Vec2,
        info: BSDFInfo,
        n: Vec3,
        radiance: Spectrum,
    }
    #[derive(Copy, Clone)]
    struct VertexTemp {
        beta: Spectrum,
        li: Spectrum,
    }
    impl CachedPathTracer {
        fn li<'a>(
            &self,
            scene: &'a Scene,
            mut ray: Ray,
            sampler: &mut dyn Sampler,
            max_depth: u32,
            path: &mut Vec<Vertex>,
            path_tmp: &mut Vec<VertexTemp>,
            training: bool,
            enable_cache: bool,
            use_cache_after: u32,
            cache: &RwLock<RadianceCache>,
        ) -> Spectrum {
            let mut li = Spectrum::zero();
            let mut beta = Spectrum::one();

            let mut depth = 0;
            path_tmp.clear();
            path.clear();
            // if training {
            //     path_tmp.push(VertexTemp {
            //         beta: Spectrum::one(),
            //         li: Spectrum::zero(),
            //     });
            //     path.push(Vertex {
            //         x: ray.o,
            //         dir: dir_to_spherical(&ray.d),
            //         radiance: Spectrum::zero(),
            //     });
            // }

            let accumulate_radiance = {
                let p_li = &mut li as *mut Spectrum;
                let p_beta = &mut beta as *mut Spectrum;
                let p_path_tmp = path_tmp as *mut Vec<VertexTemp>;
                move |l: Spectrum| unsafe {
                    *p_li += *p_beta * l;
                    if training {
                        let path_tmp = &mut *p_path_tmp;
                        for i in 0..path_tmp.len() {
                            let beta = path_tmp[i].beta;
                            path_tmp[i].li += beta * l;
                        }
                    }
                }
            };
            let accumulate_beta = {
                // let p_li = &mut li as *mut Spectrum;
                let p_beta = &mut beta as *mut Spectrum;
                let p_path_tmp = path_tmp as *mut Vec<VertexTemp>;
                move |b: Spectrum| unsafe {
                    // *p_li += *p_beta * l;
                    *p_beta *= b;
                    let path_tmp = &mut *p_path_tmp;
                    if training {
                        for i in 0..path_tmp.len() {
                            path_tmp[i].beta *= b;
                        }
                    }
                }
            };
            loop {
                if let Some(isct) = scene.shape.intersect(&ray) {
                    let ng = isct.ng;
                    let frame = Frame::from_normal(&ng);
                    let shape = isct.shape;
                    let opt_bsdf = shape.bsdf();
                    if opt_bsdf.is_none() {
                        break;
                    }
                    let p = ray.at(isct.t);
                    let bsdf = BSDFClosure {
                        frame,
                        bsdf: opt_bsdf.unwrap(),
                    };
                    if training {
                        path_tmp.push(VertexTemp {
                            beta: Spectrum::one(),
                            li: Spectrum::zero(),
                        });
                        path.push(Vertex {
                            x: p,
                            n: ng,
                            info: bsdf.bsdf.info(),
                            dir: dir_to_spherical(&ray.d),
                            radiance: Spectrum::zero(),
                        });
                    }
                    {
                        let cache_enable_depth = if training {
                            use_cache_after + 2
                        } else {
                            use_cache_after
                        };

                        if enable_cache && depth >= cache_enable_depth {
                            let cache = cache.read().unwrap();
                            let record = QueryRecord {
                                x: p,
                                dir: dir_to_spherical(&ray.d),
                                info: bsdf.bsdf.info(),
                                n: ng,
                            };
                            if let Some(radiance) = cache.infer(&record) {
                                accumulate_radiance(radiance);
                                break;
                            }
                        }
                    }

                    let wo = -ray.d;
                    if depth >= max_depth {
                        break;
                    }
                    depth += 1;
                    {
                        let (light, light_pdf) = scene.light_distr.sample(sampler.next1d());
                        let p_ref = ReferencePoint { p, n: ng };
                        let light_sample = light.sample_li(&sampler.next2d(), &p_ref);
                        if !light_sample.li.is_black()
                            && !scene.shape.occlude(&light_sample.shadow_ray)
                        {
                            // li += beta
                            //     * bsdf.evaluate(&wo, &light_sample.wi)
                            //     * glm::dot(&ng, &light_sample.wi).abs()
                            //     * light_sample.li
                            //     / (light_sample.pdf * light_pdf);
                            accumulate_radiance(
                                bsdf.evaluate(&wo, &light_sample.wi)
                                    * glm::dot(&ng, &light_sample.wi).abs()
                                    * light_sample.li
                                    / (light_sample.pdf * light_pdf),
                            );
                        }
                    }

                    if let Some(bsdf_sample) = bsdf.sample(&sampler.next2d(), &wo) {
                        let wi = &bsdf_sample.wi;
                        ray = Ray::spawn(&p, wi).offset_along_normal(&ng);
                        accumulate_beta(bsdf_sample.f * glm::dot(wi, &ng).abs() / bsdf_sample.pdf);
                    } else {
                        break;
                    }
                } else {
                    break;
                }
            }

            if training {
                for i in 0..path.len() {
                    path[i].radiance = path_tmp[i].li;
                }
            }
            li
        }
    }
    #[derive(Clone)]
    struct PerThreadData {
        path: Vec<Vertex>,
        path_tmp: Vec<VertexTemp>,
    }
    impl Integrator for CachedPathTracer {
        type Output = Film;
        fn render(&mut self, scene: &Scene) -> Self::Output {
            let npixels = (scene.camera.resolution().x * scene.camera.resolution().y) as usize;
            let film = Film::new(&scene.camera.resolution());
            let opt = SGD {
                learning_rate: 0.003,

                ..Default::default()
            };
            let cache = RwLock::new(RadianceCache::new(opt));
            let mut samplers: Vec<Box<dyn Sampler>> = vec![];
            for i in 0..npixels {
                samplers.push(Box::new(PCGSampler { rng: PCG::new(i) }));
            }
            let mut per_thread_data: Vec<PerThreadData> = vec![
                PerThreadData {
                    path: vec![],
                    path_tmp: vec![]
                };
                rayon::current_num_threads()
            ];
            for iter in 0..self.training_samples {
                let now = std::time::Instant::now();
                let p_samplers = &UnsafePointer::new(&mut samplers as *mut Vec<Box<dyn Sampler>>);
                let p_per_thread_data =
                    &UnsafePointer::new(&mut per_thread_data as *mut Vec<PerThreadData>);
                parallel_for(npixels, 256, |id| {
                    let samplers = unsafe { p_samplers.as_mut().unwrap() };
                    let thread_data = unsafe {
                        &mut (p_per_thread_data.as_mut().unwrap())
                            [rayon::current_thread_index().unwrap()]
                    };
                    let sampler = &mut samplers[id];
                    // let mut sampler = PCGSampler { rng: PCG::new(id) };
                    let x = (id as u32) % scene.camera.resolution().x;
                    let y = (id as u32) / scene.camera.resolution().x;
                    let pixel = uvec2(x, y);
                    let (ray, _ray_weight) = scene.camera.generate_ray(&pixel, sampler.as_mut());
                    let path = &mut thread_data.path;
                    let path_tmp = &mut thread_data.path_tmp;
                    let training = ((x + iter) % 8 == 0) && ((y + iter / 8) % 8 == 0);
                    if !training {
                        return;
                    }
                    let _li = self.li(
                        scene,
                        ray,
                        sampler.as_mut(),
                        self.max_depth,
                        path,
                        path_tmp,
                        training,
                        iter >= 128,
                        2,
                        &cache,
                    );
                    if training {
                        let mut lk = cache.write().unwrap();
                        let cache = &mut *lk;
                        for vertex in path.iter() {
                            // vertex.dir.into_iter().for_each(|x| assert!(!x.is_nan()));
                            let record = QueryRecord {
                                x: vertex.x,
                                dir: vertex.dir,
                                info: vertex.info,
                                n: vertex.n,
                            };
                            cache.train(&record, &vertex.radiance);
                        }
                    }
                });
                println!(
                    "training pass {} finished in {}s",
                    iter,
                    now.elapsed().as_secs_f32()
                );
            }
            // println!("visiualizing");
            for _iter in 0..self.spp {
                let p_samplers = &UnsafePointer::new(&mut samplers as *mut Vec<Box<dyn Sampler>>);
                let p_per_thread_data =
                    &UnsafePointer::new(&mut per_thread_data as *mut Vec<PerThreadData>);
                parallel_for(npixels, 256, |id| {
                    let samplers = unsafe { p_samplers.as_mut().unwrap() };
                    let thread_data = unsafe {
                        &mut (p_per_thread_data.as_mut().unwrap())
                            [rayon::current_thread_index().unwrap()]
                    };
                    let sampler = &mut samplers[id];
                    // let mut sampler = PCGSampler { rng: PCG::new(id) };
                    let x = (id as u32) % scene.camera.resolution().x;
                    let y = (id as u32) / scene.camera.resolution().x;
                    let pixel = uvec2(x, y);
                    let (ray, _ray_weight) = scene.camera.generate_ray(&pixel, sampler.as_mut());
                    let path = &mut thread_data.path;
                    let path_tmp = &mut thread_data.path_tmp;
                    let training = false;
                    let li = self.li(
                        scene,
                        ray,
                        sampler.as_mut(),
                        self.max_depth,
                        path,
                        path_tmp,
                        training,
                        true,
                        1,
                        &cache,
                    );
                    // let li = {
                    //     if let Some(isct) = scene.shape.intersect(&ray) {
                    //         let p = ray.at(isct.t);
                    //         let n = isct.ng;
                    //         let bsdf = isct.shape.bsdf();
                    //         if bsdf.is_none() {
                    //             Spectrum::zero()
                    //         } else {
                    //             let bsdf = bsdf.unwrap();
                    //             let record = QueryRecord {
                    //                 x: p,
                    //                 n,
                    //                 info: bsdf.info(),
                    //                 dir: dir_to_spherical(&ray.d),
                    //             };
                    //             let cache = cache.read().unwrap();
                    //             cache.infer(&record).unwrap_or(Spectrum::zero())
                    //         }
                    //     } else {
                    //         Spectrum::zero()
                    //     }
                    // };
                    film.add_sample(&uvec2(x, y), &li, 1.0);
                });
            }
            // println!("visiualizing");
            // parallel_for(npixels, 256, |id| {
            //     let mut sampler = PCGSampler { rng: PCG::new(id) };
            //     let x = (id as u32) % scene.camera.resolution().x;
            //     let y = (id as u32) / scene.camera.resolution().x;
            //     let pixel = uvec2(x, y);
            //     let (ray, _ray_weight) = scene.camera.generate_ray(&pixel, &mut sampler);
            //     let lk = cache.read().unwrap();
            //     let cache = &*lk;
            //     // let li = cache.infer(&ray.o, &dir_to_spherical(&ray.d));
            //     let mut li = Spectrum::zero();
            //     if let Some(isct) = scene.shape.intersect(&ray) {
            //         let p = ray.o; //ray.at(isct.t * 0.9);
            //         li = cache.infer(&p, &dir_to_spherical(&ray.d));
            //         // let n = isct.ng;
            //         // let frame = Frame::from_normal(&n);
            //         // f
            //         // println!("{}", li.samples);
            //     }
            //     film.add_sample(&uvec2(x, y), &li, 1.0);
            // });
            film
        }
    }
}

#[cfg(test)]
mod tests {
    // use nn::mlp
    use super::nnv2::*;
    use nalgebra as na;
    use rand::Rng;
    #[test]
    fn test_mlp() {
        // #[macro_use(nn::mlp)]

        let opt_params = AdamParams {
            learning_rate: 0.003,
            ..Default::default()
        };
        let opt = Adam::new(opt_params);
        // let mut net = Model::new(sequential!(layer0, layer1));
        let mut layers = vec![];
        {
            layers.push(Layer::new::<Relu>(2, 8));
            layers.push(Layer::no_activation(8, 1));
        }
        let mut net = MLP::new(layers, opt);

        fn f(x: &na::DMatrix<f32>) -> na::DMatrix<f32> {
            // na::SVector::<f32, 1>::new(x[0] + x[1])
            na::DMatrix::<f32>::from_fn(1, x.ncols(), |r, c| x[(0, c)] + x[(1, c)])
        }
        let mut rng = rand::thread_rng();
        let batch_size = 1024;
        for iter in 0..100000 {
            // let x = na::zero::<na::SVector<f32, 2>>().map(|_| rng.gen::<f32>() * 2.0 - 1.0);
            let x =
                na::DMatrix::<f32>::from_fn(2, batch_size, |_r, _c| rng.gen::<f32>() * 2.0 - 1.0);
            // let y = net.infer(&x);
            let loss = net.train(x.clone(), &f(&x));
            if iter % 1000 == 0 {
                // println!("training on {} {}; {}",&x, f(&x), y);
                println!("{} {}", iter, loss);
            }
        }
        {
            let x = na::DMatrix::<f32>::from_row_slice(2, 1, &[0.1, 0.2]);
            let y = net.infer(x.clone());
            let gt = f(&x);
            let err: f32 = (&y - &gt).norm();
            println!("y:{}, gt:{}, err = {}", y, gt, err);
            assert!(err < 0.001);
            assert!(!err.is_nan());
        }
        // println!("{}", net)
        // println!("{} {}", net.infer(& na::SVector::<f32, 1>::new(0.3f32)), f(& na::SVector::<f32, 1>::new(0.3f32)));
        // println!("{} {}", net.m.a.linear.weights, net.m.a.linear.bias);
        // println!("{} {}", net.m.b.weights, net.m.b.bias);
    }
}
mod test_image {
    // use nn::mlp
    use super::nnv2::*;
    use image::GenericImageView;
    use nalgebra as na;
    use rand::Rng;
    // create_mlp!(Net, Relu, SGD, { 2 * 8 * 2 + 2 }, 64, 64, 64, 64, 64, 3);
    use crate::position_encoding_func_v2;
    position_encoding_func_v2!(position_encoding, 2, 8);

    pub fn test() {
        let opt_params = AdamParams {
            learning_rate: 0.003,
            ..Default::default()
        };
        let opt = Adam::new(opt_params);
        let mut layers = vec![];
        {
            layers.push(Layer::new::<Relu>(2 * 8 * 2 + 2, 64));
            layers.push(Layer::new::<Relu>(64, 64));
            layers.push(Layer::new::<Relu>(64, 64));
            layers.push(Layer::new::<Relu>(64, 64));
            layers.push(Layer::new::<Relu>(64, 64));
            layers.push(Layer::new::<Relu>(64, 64));
            layers.push(Layer::no_activation(64, 3));
        }
        let mut net = MLP::new(layers, opt);
        let img = image::open("test.jpg").unwrap();
        let imgx = img.width();
        let imgy = img.height();
        let get_pixel = |x: &na::DMatrix<f32>| -> na::DMatrix<f32> {
            let mut pixels: na::DMatrix<f32> = na::DMatrix::zeros(3, x.ncols());
            for c in 0..x.ncols() {
                let x = x.column(c);
                let px = img.get_pixel((x[0] * imgx as f32) as u32, (x[1] * imgy as f32) as u32);
                for i in 0..3 {
                    pixels[(i, c)] = px[i] as f32 / 255.0;
                }
            }
            pixels
        };
        let mut rng = rand::thread_rng();
        let batch_size = 5000;
        for iter in 0..1000 {
            let x: na::DMatrix<f32> = na::DMatrix::from_fn(2, batch_size, |r, c|->f32 {rng.gen::<f32>()});
            let target = get_pixel(&x);
            let loss = net.train(position_encoding(&x),  &target);
            // if iter % 1000 == 0 {
                // println!("training on {} {}; {}",&x, f(&x), y);
                println!("{} {}", iter, loss);
            // }
        }
        {
            let x = na::DMatrix::<f32>::from_fn(2, (imgx * imgy) as usize, |r, c| {
                let x = c as u32 % imgx;
                let y = c as u32 / imgx;
                if r == 0 {
                    x as f32 / imgx as f32
                } else {
                    y as f32 / imgy as f32
                }
            });
            let color = net.infer(position_encoding(&x)) * 255.0;
            let out = image::ImageBuffer::from_fn(imgx, imgy, |x, y| {
                let i = x + imgx * y;
                let pixel = [
                    color[(0, i as usize)] as u8,
                    color[(1, i as usize)] as u8,
                    color[(2, i as usize)] as u8,
                ];
                image::Rgb([pixel[0] as u8, pixel[1] as u8, pixel[2] as u8])
            });
            out.save("out.jpg").unwrap();
        }
        // let out = image::ImageBuffer::from_fn(imgx, imgy, |x, y| {
        //     let x:na::DMatrix<f32> = na::Vector2::<f32>::new(x as f32 / imgx as f32, y as f32 / imgy as f32);
        //     let color = net.infer(position_encoding(&x)) * 255.0;
        //     image::Rgb([color[0] as u8, color[1] as u8, color[2] as u8])
        // });
    }
}
fn main() {
    test_image::test();
}

#[macro_use]
mod nnv2 {
    use std::{cell::RefCell, collections::LinkedList, fmt::Pointer, rc::Rc};

    use nalgebra as na;
    #[derive(Clone)]
    struct Dual<T> {
        pub val: T,
        pub grad: T,
    }
    type MatrixXf = na::DMatrix<f32>;
    type VectorXf = na::DVector<f32>;
    pub trait Activation {
        fn new() -> Self
        where
            Self: Sized;
        fn forward(&self, x: &MatrixXf) -> MatrixXf;
        fn backward(&self, x: &MatrixXf, out: Dual<&MatrixXf>) -> MatrixXf;
    }
    pub struct Relu {}
    impl Activation for Relu {
        fn new() -> Self
        where
            Self: Sized,
        {
            Self {}
        }
        fn forward(&self, x: &MatrixXf) -> MatrixXf {
            x.map(|v| v.max(0.0))
        }
        fn backward(&self, x: &MatrixXf, out: Dual<&MatrixXf>) -> MatrixXf {
            MatrixXf::from_fn(x.nrows(), x.ncols(), |r, c| {
                if x[(r, c)] > 0.0 {
                    out.grad[(r, c)]
                } else {
                    0.0
                }
            })
        }
    }
    pub struct Layer {
        pub inputs: usize,
        pub outputs: usize,
        pub activation: Option<Box<dyn Activation>>,
    }
    impl Layer {
        pub fn new<A: Activation + 'static>(inputs: usize, outputs: usize) -> Self {
            Layer {
                inputs,
                outputs,
                activation: Some(Box::new(A::new()) as Box<dyn Activation>),
            }
        }
        pub fn no_activation(inputs: usize, outputs: usize) -> Self {
            Layer {
                inputs,
                outputs,
                activation: None,
            }
        }
    }
    pub trait Optimizer: Clone {
        fn create(&self, n: usize) -> Box<dyn OptimizerImpl>;
    }
    trait OptimizerImpl {
        fn step(&mut self, val: &mut [f32], grad: &[f32]);
    }

    #[derive(Copy, Clone)]
    pub struct SGDParams {
        pub learning_rate: f32,
    }
    impl Default for SGDParams {
        fn default() -> Self {
            Self {
                learning_rate: 0.003,
            }
        }
    }
    #[derive(Clone)]
    pub struct SGD {
        params: Rc<RefCell<SGDParams>>,
    }
    impl SGD {
        pub fn new(params: SGDParams) -> Self {
            Self {
                params: Rc::new(RefCell::new(params)),
            }
        }
    }
    struct SGDImpl {
        params: Rc<RefCell<SGDParams>>,
    }
    impl Optimizer for SGD {
        fn create(&self, _n: usize) -> Box<dyn OptimizerImpl> {
            Box::new(SGDImpl {
                params: self.params.clone(),
            })
        }
    }
    impl OptimizerImpl for SGDImpl {
        fn step(&mut self, val: &mut [f32], grad: &[f32]) {
            let lr = self.params.borrow().learning_rate;
            assert!(val.len() == grad.len());
            for i in 0..val.len() {
                val[i] -= lr * grad[i].min(100.0).max(-100.0);
            }
        }
    }
    #[derive(Clone, Copy)]
    pub struct AdamParams {
        pub learning_rate: f32,
        pub beta1: f32,
        pub beta2: f32,
    }
    impl Default for AdamParams {
        fn default() -> Self {
            Self {
                learning_rate: 0.001,
                beta1: 0.9,
                beta2: 0.999,
            }
        }
    }
    #[derive(Clone)]
    pub struct Adam {
        params: Rc<RefCell<AdamParams>>,
    }
    impl Adam {
        pub fn new(params: AdamParams) -> Self {
            Self {
                params: Rc::new(RefCell::new(params)),
            }
        }
    }
    impl Optimizer for Adam {
        fn create(&self, n: usize) -> Box<dyn OptimizerImpl> {
            Box::new(AdamImpl {
                params: self.params.clone(),
                m: vec![0.0; n],
                v: vec![0.0; n],
                t: 0,
            })
        }
    }
    struct AdamImpl {
        params: Rc<RefCell<AdamParams>>,
        m: Vec<f32>,
        v: Vec<f32>,
        t: i32,
    }
    impl OptimizerImpl for AdamImpl {
        fn step(&mut self, val: &mut [f32], grad: &[f32]) {
            let params = self.params.borrow();
            let lr = params.learning_rate;
            let beta1 = params.beta1;
            let beta2 = params.beta2;
            self.t += 1;

            for i in 0..val.len() {
                let grad = grad[i].min(100.0).max(-100.0);
                self.m[i] = beta1 * self.m[i] + (1.0 - beta1) * grad;
                self.v[i] = beta2 * self.v[i] + (1.0 - beta2) * grad * grad;
            }
            let b1 = beta1.powi(self.t);
            let b2 = beta2.powi(self.t);
            for i in 0..val.len() {
                let m_tilde = self.m[i] / (1.0 - b1);
                let v_tilde = self.v[i] / (1.0 - b2);
                val[i] -= lr * (m_tilde / (v_tilde.sqrt() + 1e-8f32)) as f32;
            }
        }
    }
    // struct Optimizer
    struct LayerData {
        weights: MatrixXf,
        bias: VectorXf,
        activation: Option<Box<dyn Activation>>,
    }
    struct LayerOptimizer {
        weights: Box<dyn OptimizerImpl>,
        bias: Box<dyn OptimizerImpl>,
    }
    use rand::distributions::Distribution;
    use statrs::distribution::Normal;
    struct LayerOutput {
        linear_out: MatrixXf,
        out: MatrixXf,
    }
    impl LayerData {
        fn num_inputs(&self) -> usize {
            self.weights.ncols()
        }
        fn num_outputs(&self) -> usize {
            self.weights.nrows()
        }
        fn new(inputs: usize, outputs: usize, activation: Option<Box<dyn Activation>>) -> Self {
            let mut rng = rand::thread_rng();
            let n = Normal::new(0.0, (2.0 / (inputs + outputs) as f64).sqrt()).unwrap();
            let weights = MatrixXf::from_fn(outputs, inputs, |_r, _c| n.sample(&mut rng) as f32);
            let bias = VectorXf::from_fn(outputs, |_r, _c| 0.0f32);
            Self {
                weights,
                bias,
                activation,
            }
        }
    }
    pub struct MLP {
        layers: Vec<LayerData>,
        opts: Vec<LayerOptimizer>,
    }
    impl MLP {
        pub fn new<O: Optimizer>(desc: Vec<Layer>, opt: O) -> Self {
            let mut layers = vec![];
            let mut opts = vec![];
            for d in desc {
                layers.push(LayerData::new(d.inputs, d.outputs, d.activation));
                opts.push(LayerOptimizer {
                    weights: opt.create(d.inputs * d.outputs),
                    bias: opt.create(d.outputs),
                })
            }
            Self { layers, opts }
        }
        fn forward(&self, mut x: MatrixXf, tmp: &mut Vec<LayerOutput>, training: bool) -> MatrixXf {
            for layer in &self.layers {
                x = &layer.weights * x;
                x = MatrixXf::from_fn(x.nrows(), x.ncols(), |r, c| x[(r, c)] + layer.bias[r]);
                let linear_out = if training { Some(x.clone()) } else { None };
                if let Some(f) = &layer.activation {
                    x = f.forward(&x);
                }
                if training {
                    tmp.push(LayerOutput {
                        linear_out: linear_out.unwrap(),
                        out: x.clone(),
                    })
                }
            }
            x
        }
        pub fn infer(&self, x: MatrixXf) -> MatrixXf {
            let mut tmp = vec![];
            self.forward(x, &mut tmp, false)
        }
        pub fn train(&mut self, x: MatrixXf, target: &MatrixXf) -> f32 {
            let mut tmp = vec![];
            let y = self.forward(x.clone(), &mut tmp, true);
            let loss = (&y - target).map(|v| v * v).mean();
            let dy = 0.5 * (&y - target) / (y.ncols() * y.nrows()) as f32;
            let mut dout = dy;
            for i in (0..self.layers.len()).rev() {
                let layer = &self.layers[i];
                let out = &tmp[i].out;
                let linear_out = &tmp[i].linear_out;
                if let Some(f) = &layer.activation {
                    let gradf = f.backward(
                        linear_out,
                        Dual {
                            val: out,
                            grad: &dout,
                        },
                    );
                    // out = linear_out;
                    dout = gradf;
                }
                let input = if i == 0 { &x } else { &tmp[i - 1].out };
                let (dw, dbias, dx) = {
                    let dbias = dout.column_sum();
                    let dw = &dout * input.transpose();
                    let dx = &layer.weights.transpose() * &dout;
                    (dw, dbias, dx)
                };
                let opt = &mut self.opts[i];
                opt.weights
                    .as_mut()
                    .step(self.layers[i].weights.as_mut_slice(), dw.as_slice());
                opt.bias
                    .as_mut()
                    .step(self.layers[i].bias.as_mut_slice(), dbias.as_slice());
                dout = dx;
            }
            loss
        }
    }
    #[macro_export]
    macro_rules! position_encoding_func_v2 {
        ($name:ident, $N:expr, $E:expr) => {
            fn $name(v: &na::DMatrix<f32>) -> na::DMatrix<f32> {
                assert!(v.nrows() == $N);
                let mut u: na::DMatrix<f32> = na::DMatrix::zeros($N * $E * 2 + $N, v.ncols());
                for c in 0..v.ncols() {
                    for i in 0..$N {
                        for j in 0..$E {
                            let feq = 2.0f32.powi(j as i32);
                            u[(i * $E + j, c)] = (v[(i, c)] * feq).sin();
                            u[(i * $E + j + $N * $E, c)] = (v[(i, c)] * feq).cos();
                        }
                        u[($N * $E * 2 + i, c)] = v[(i, c)];
                    }
                }
                u
            }
        };
    }
}
fn gemm_bench1() {
    let a = vec![na::DMatrix::<f32>::from_fn(64, 64, |_r, _c| { 1.0 }); 14];
    let b = na::DMatrix::<f32>::from_fn(64, 64, |_r, _c| 1.0);
    let now = std::time::Instant::now();
    let mut c = &a[13] * &b;
    for i in (0..13).rev() {
        c = &a[i] * c;
    }
    let tmp = c.sum();
    let t = now.elapsed().as_secs_f64();
    // let flops = 2.0 * (a[0].nrows() * a[0].ncols() * b.ncols()) as f64 / t / 1e9;
    println!("{}s {}", t, tmp);
}

fn gemm_bench2() {
    let a = vec![na::DMatrix::<f32>::from_fn(64, 64, |_r, _c| { 1.0 }); 14];
    let mut b = na::SMatrix::<f32, 64, 1>::from_fn(|_r, _c| 1.0);
    let mut tmp = 0.0;
    let now = std::time::Instant::now();
    for _ in 0..64 {
        let mut c = &a[13] * &b;
        for i in (0..13).rev() {
            c = &a[i] * c;
        }
        tmp += c.sum();
    }
    let t = now.elapsed().as_secs_f64();
    // let flops = 2.0 * (a[0].nrows() * a[0].ncols() * 1024) as f64 / t / 1e9;
    println!("{}s {}", t, tmp);
}

fn main1() {
    // rayon::ThreadPoolBuilder::new()
    //     .num_threads(1)
    //     .build_global()
    //     .unwrap();
    let white = Arc::new(DiffuseBSDF {
        reflecance: Spectrum::one(),
    });
    let red = Arc::new(DiffuseBSDF {
        reflecance: Spectrum::from_srgb(&vec3(0.75, 0.25, 0.25)),
    });
    let green = Arc::new(DiffuseBSDF {
        reflecance: Spectrum::from_srgb(&vec3(0.25, 0.75, 0.25)),
    });
    let shape = {
        let mut shapes: Vec<Arc<dyn Shape>> = vec![];
        shapes.push(Arc::new(Sphere {
            center: vec3(0.0, 0.0, -4.0),
            radius: 1.0,
            bsdf: white.clone(),
        }));
        shapes.push(Arc::new(Sphere {
            center: vec3(10000.0 + 4.0, 0.0, -0.0),
            radius: 10000.0,
            bsdf: red.clone(),
        }));
        shapes.push(Arc::new(Sphere {
            center: vec3(-10000.0 - 4.0, 0.0, -0.0),
            radius: 10000.0,
            bsdf: green.clone(),
        }));
        shapes.push(Arc::new(Sphere {
            center: vec3(0.0, -10000.0 - 1.0, -0.0),
            radius: 10000.0,
            bsdf: white.clone(),
        }));
        shapes.push(Arc::new(Sphere {
            center: vec3(0.0, 10000.0 + 6.0, -0.0),
            radius: 10000.0,
            bsdf: white.clone(),
        }));
        shapes.push(Arc::new(Sphere {
            center: vec3(0.0, 0.0, -10015.0),
            radius: 10000.0,
            bsdf: white.clone(),
        }));
        Arc::new(Aggregate::new(shapes))
    };
    let camera = {
        let m = glm::translate(&glm::identity(), &vec3(0.0, 0.4, 0.0));
        Arc::new(PerspectiveCamera::new(
            &uvec2(256, 256),
            &Transform::from_matrix(&m),
            (80.0 as Float).to_radians(),
        ))
    };
    let lights: Vec<Arc<dyn Light>> = vec![Arc::new(PointLight {
        emission: Spectrum::one() * 40.0,
        position: vec3(0.3, 4.0, 0.0),
    })];
    let scene = Scene {
        shape,
        camera,
        lights: lights.clone(),
        light_distr: Arc::new(UniformLightDistribution::new(lights.clone())),
    };
    // let mut integrator = PathTracer {
    //     spp: 16,
    //     max_depth: 3,
    // };
    let mut integrator = nrc::CachedPathTracer {
        spp: 16,
        training_samples: 256,
        max_depth: 3,
    };
    // let mut integrator = BDPT {
    //     spp: 32,
    //     max_depth: 3,
    //     debug: false,
    // };
    // let mut integrator = SPPM {
    //     initial_radius: 0.1,
    //     iterations: 64,
    //     max_depth: 5,
    //     n_photons: 100000,
    // };
    let film = integrator.render(&scene);
    let image = film.to_rgb_image();
    image.save("out-nrc.png").unwrap();
}
