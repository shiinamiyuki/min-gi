pub use nalgebra as na;
pub use nalgebra_glm as glm;
use rayon::prelude::*;
use std::{
    alloc::dealloc,
    cell::{RefCell, UnsafeCell},
    collections::{HashMap, LinkedList},
    mem::MaybeUninit,
    ops::{Deref, Index, IndexMut, Mul},
    sync::{
        atomic::{AtomicPtr, AtomicU32, Ordering},
        Arc, RwLock,
    },
    usize,
};

type Float = f64;
type Vec3 = glm::TVec3<Float>;
type Vec2 = glm::TVec2<Float>;
type Mat4 = glm::TMat4<Float>;
type Mat3 = glm::TMat3<Float>;

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
}
impl Default for Bound3<Float> {
    fn default() -> Self {
        let inf = Float::INFINITY;
        Self {
            min: Vec3::new(inf, inf, inf),
            max: Vec3::new(-inf, -inf, -inf),
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
            samples: Vec3::new(f(rgb.x), f(rgb.y), f(rgb.z)),
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

        Vec3::new(f(self.samples.x), f(self.samples.y), f(self.samples.z))
    }
    pub fn zero() -> Spectrum {
        Self {
            samples: glm::zero(),
        }
    }
    pub fn one() -> Spectrum {
        Self {
            samples: Vec3::new(1.0, 1.0, 1.0),
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
            glm::normalize(&Vec3::new(-normal.z, 0.0, normal.x))
        } else {
            glm::normalize(&Vec3::new(0.0, normal.z, -normal.y))
        };
        Self {
            N: *normal,
            T: tangent,
            B: glm::normalize(&glm::cross(normal, &tangent)),
        }
    }
    pub fn to_local(&self, v: &Vec3) -> Vec3 {
        Vec3::new(
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
        Vec3::new(q.x, q.y, q.z) / q.w
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
    let u_offset: Vec2 = 2.0 * u - Vec2::new(1.0, 1.0);
    if u_offset.x == 0.0 && u_offset.y == 0.0 {
        return Vec2::new(0.0, 0.0);
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
    r * Vec2::new(theta.cos(), theta.sin())
}
pub fn consine_hemisphere_sampling(u: &Vec2) -> Vec3 {
    let uv = concentric_sample_disk(&u);
    let r = glm::dot(&uv, &uv);
    let h = (1.0 - r).sqrt();
    Vec3::new(uv.x, h, uv.y)
}
pub fn uniform_sphere_sampling(u: &Vec2) -> Vec3 {
    let z = 1.0 - 2.0 * u[0];
    let r = (1.0 - z * z).max(0.0).sqrt();
    let phi = 2.0 * PI * u[1];
    Vec3::new(r * phi.cos(), z, r * phi.sin())
}
pub fn uniform_sphere_pdf() -> Float {
    1.0 / (4.0 * PI)
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
pub trait BSDF: Sync + Send {
    fn evaluate(&self, wo: &Vec3, wi: &Vec3) -> Spectrum;
    fn evaluate_pdf(&self, wo: &Vec3, wi: &Vec3) -> Float;
    fn sample(&self, u: &Vec2, wo: &Vec3) -> Option<BSDFSample>;
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
        Vec2::new(self.next1d(), self.next1d())
    }
}

pub trait Camera: Sync + Send {
    fn generate_ray(&self, pixel: &glm::UVec2, sampler: &mut dyn Sampler) -> Ray;
    fn resolution(&self) -> glm::UVec2;
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
}
#[derive(Clone, Copy)]
pub struct ReferencePoint {
    pub p: Vec3,
    pub n: Vec3,
}
pub trait Light: Sync + Send {
    fn sample_le(&self, u: &[Vec2; 2]) -> LightRaySample;
    fn sample_li(&self, u: &Vec2, p: &ReferencePoint) -> LightSample;
    fn le(&self, ray: &Ray) -> Spectrum;
}

#[derive(Clone)]
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
#[derive(Clone)]
struct Pixel {
    intensity: Spectrum,
    weight: Float,
}
struct Film {
    pixels: Vec<Pixel>,
    resolution: glm::UVec2,
}
impl Film {
    pub fn new(resolution: &glm::UVec2) -> Self {
        Self {
            pixels: vec![
                Pixel {
                    intensity: Spectrum::zero(),
                    weight: 0.0
                };
                (resolution.x * resolution.y) as usize
            ],
            resolution: *resolution,
        }
    }
    pub fn add_sample(&mut self, pixel: &glm::UVec2, value: &Spectrum, weight: Float) {
        let pixel = &mut self.pixels[(pixel.x + pixel.y * self.resolution.x) as usize];
        pixel.intensity = pixel.intensity + *value;
        pixel.weight += weight;
    }
    pub fn get_pixel(&self, pixel: &glm::UVec2) -> &Pixel {
        &self.pixels[(pixel.x + pixel.y * self.resolution.x) as usize]
    }
    pub fn to_rgb_image(&self) -> image::RgbImage {
        let image = image::ImageBuffer::from_fn(self.resolution.x, self.resolution.y, |x, y| {
            let pixel = self.get_pixel(&glm::UVec2::new(x, y));
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
            let invd: Vec3 = Vec3::new(1.0, 1.0, 1.0).component_div(&ray.d);
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
            let invd: Vec3 = Vec3::new(1.0, 1.0, 1.0).component_div(&ray.d);
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
                uv: Vec2::new(0.0, 0.0),
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
                uv: Vec2::new(0.0, 0.0),
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
            min: self.center - Vec3::new(self.radius, self.radius, self.radius),
            max: self.center + Vec3::new(self.radius, self.radius, self.radius),
        }
    }
}
struct PerspectiveCamera {
    resolution: glm::UVec2,
    c2w: Transform,
    w2c: Transform,
    fov: Float,
    r2c: Transform,
    c2r: Transform,
}
impl PerspectiveCamera {
    fn new(resolution: &glm::UVec2, transform: &Transform, fov: Float) -> Self {
        let mut m = glm::identity();
        let fres = Vec2::new(resolution.x as Float, resolution.y as Float);
        m = glm::scale(
            &glm::identity(),
            &Vec3::new(1.0 / fres.x, 1.0 / fres.y, 1.0),
        ) * m;
        m = glm::scale(&glm::identity(), &Vec3::new(2.0, 2.0, 1.0)) * m;
        m = glm::translate(&glm::identity(), &Vec3::new(-1.0, -1.0, 0.0)) * m;
        m = glm::scale(&glm::identity(), &Vec3::new(1.0, -1.0, 1.0)) * m;
        let s = (fov / 2.0).atan();
        if resolution.x > resolution.y {
            m = glm::scale(&glm::identity(), &Vec3::new(s, s * fres.y / fres.x, 1.0)) * m;
        } else {
            m = glm::scale(&glm::identity(), &Vec3::new(s * fres.x / fres.y, s, 1.0)) * m;
        }
        let r2c = Transform::from_matrix(&m);
        Self {
            resolution: *resolution,
            c2w: *transform,
            w2c: transform.inverse().unwrap(),
            r2c,
            c2r: r2c.inverse().unwrap(),
            fov,
        }
    }
}
impl Camera for PerspectiveCamera {
    fn generate_ray(&self, pixel: &glm::UVec2, sampler: &mut dyn Sampler) -> Ray {
        // let p_lens = consine_hemisphere_sampling(&sampler.next2d())
        let fpixel: Vec2 = glm::convert(*pixel);
        let p_film = sampler.next2d() + fpixel;
        let p = {
            let v = self
                .r2c
                .transform_point(&Vec3::new(p_film.x, p_film.y, 0.0));
            Vec2::new(v.x, v.y)
        };
        let mut ray = Ray::spawn(
            &glm::zero(),
            &glm::normalize(&(Vec3::new(p.x, p.y, 0.0) - Vec3::new(0.0, 0.0, 1.0))),
        );
        ray.o = self.c2w.transform_point(&ray.o);
        ray.d = self.c2w.transform_vector(&ray.d);
        ray
    }
    fn resolution(&self) -> glm::UVec2 {
        self.resolution
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
        }
    }
    fn le(&self, _: &Ray) -> Spectrum {
        unimplemented!("point light cannot be hit")
    }
}
struct DiffuseBSDF {
    reflecance: Spectrum,
}
impl BSDF for DiffuseBSDF {
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
                Vec3::new(w.x, -w.y, w.z)
            }
        };
        Some(BSDFSample {
            f: self.reflecance * FRAC_1_PI,
            wi,
            pdf: Frame::abs_cos_theta(&wi) * FRAC_1_PI,
        })
    }
}
struct Scene {
    shape: Arc<dyn Shape>,
    camera: Arc<dyn Camera>,
    lights: Vec<Arc<dyn Light>>,
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
            let pixel = glm::UVec2::new(x, y);
            let mut acc_li = Spectrum::zero();
            for _ in 0..self.spp {
                let mut ray = scene.camera.generate_ray(&pixel, &mut sampler);
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
                        //     samples: Vec3::new(1.0,1.0,1.0) * glm::dot(&wi, &Vec3::new(0.2, 0.8, 0.0)),
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
                film.add_sample(&glm::UVec2::new(x, y), &acc_li, 1.0);
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
        let film = RwLock::new(Film::new(&scene.camera.resolution()));
        parallel_for(npixels, 256, |id| {
            let mut sampler = PCGSampler { rng: PCG::new(id) };
            let x = (id as u32) % scene.camera.resolution().x;
            let y = (id as u32) / scene.camera.resolution().x;
            let pixel = glm::UVec2::new(x, y);
            let mut acc_li = Spectrum::zero();
            for _ in 0..self.spp {
                let mut ray = scene.camera.generate_ray(&pixel, &mut sampler);
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
                                let light = &scene.lights[0];
                                let p_ref = ReferencePoint { p, n: ng };
                                let light_sample = light.sample_li(&sampler.next2d(), &p_ref);
                                if !light_sample.li.is_black()
                                    && !scene.shape.occlude(&light_sample.shadow_ray)
                                {
                                    li += beta
                                        * bsdf.evaluate(&wo, &light_sample.wi)
                                        * glm::dot(&ng, &light_sample.wi).abs()
                                        * light_sample.li
                                        / light_sample.pdf;
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
            {
                let film = &mut film.write().unwrap();
                film.add_sample(&glm::UVec2::new(x, y), &acc_li, 1.0);
            }
        });
        film.into_inner().unwrap()
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
        q = q.component_mul(&Vec3::new(
            self.grid_res[0] as Float,
            self.grid_res[1] as Float,
            self.grid_res[2] as Float,
        ));
        glm::UVec3::new(q.x as u32, q.y as u32, q.z as u32)
    }
    pub fn insert(&self, pixel: &'a SPPMPixel<'a>) {
        // println!("fuck");
        let p = pixel.vp.as_ref().unwrap().p;
        let radius = pixel.radius;
        let pmin = self.to_grid(p - Vec3::new(radius, radius, radius));
        let pmax = self.to_grid(p + Vec3::new(radius, radius, radius));
        // println!("{:?} {:?}", pmin,pmax);
        for z in pmin.z..=pmax.z {
            for y in pmin.y..=pmax.y {
                for x in pmin.x..=pmax.x {
                    let h = self.hash(&glm::UVec3::new(x, y, z));

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
        let film = RwLock::new(Film::new(&scene.camera.resolution()));
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
                let pixel = glm::UVec2::new(x, y);

                let sampler = unsafe { p_samplers.offset(id as isize).as_mut().unwrap().as_mut() };
                let mut ray = scene.camera.generate_ray(&pixel, sampler);
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
                            min: vp.p - Vec3::new(pixel.radius, pixel.radius, pixel.radius),
                            max: vp.p + Vec3::new(pixel.radius, pixel.radius, pixel.radius),
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
                                // if depth == 1{
                                // println!("fuck");
                                // }
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
            let pixel = glm::UVec2::new(x, y);
            let p = unsafe { p_pixels.offset(id as isize).as_ref().unwrap() };
            let l =
                p.tau / ((self.iterations * self.n_photons) as Float * PI * p.radius * p.radius);
            {
                let film = &mut film.write().unwrap();
                film.add_sample(&pixel, &l, 1.0);
            }
        });
        film.into_inner().unwrap()
    }
}

fn main() {
    let white = Arc::new(DiffuseBSDF {
        reflecance: Spectrum::one(),
    });
    let red = Arc::new(DiffuseBSDF {
        reflecance: Spectrum::from_srgb(&Vec3::new(0.75, 0.25, 0.25)),
    });
    let green = Arc::new(DiffuseBSDF {
        reflecance: Spectrum::from_srgb(&Vec3::new(0.25, 0.75, 0.25)),
    });
    let shape = {
        let mut shapes: Vec<Arc<dyn Shape>> = vec![];
        shapes.push(Arc::new(Sphere {
            center: Vec3::new(0.0, 0.0, -4.0),
            radius: 1.0,
            bsdf: white.clone(),
        }));
        shapes.push(Arc::new(Sphere {
            center: Vec3::new(10000.0 + 4.0, 0.0, -0.0),
            radius: 10000.0,
            bsdf: red.clone(),
        }));
        shapes.push(Arc::new(Sphere {
            center: Vec3::new(-10000.0 - 4.0, 0.0, -0.0),
            radius: 10000.0,
            bsdf: green.clone(),
        }));
        shapes.push(Arc::new(Sphere {
            center: Vec3::new(0.0, -10000.0 - 1.0, -0.0),
            radius: 10000.0,
            bsdf: white.clone(),
        }));
        shapes.push(Arc::new(Sphere {
            center: Vec3::new(0.0, 10000.0 + 6.0, -0.0),
            radius: 10000.0,
            bsdf: white.clone(),
        }));
        shapes.push(Arc::new(Sphere {
            center: Vec3::new(0.0, 0.0, -10015.0),
            radius: 10000.0,
            bsdf: white.clone(),
        }));
        Arc::new(Aggregate::new(shapes))
    };
    let camera = {
        let m = glm::translate(&glm::identity(), &Vec3::new(0.0, 0.4, 0.0));
        Arc::new(PerspectiveCamera::new(
            &glm::UVec2::new(512, 512),
            &Transform::from_matrix(&m),
            (80.0 as Float).to_radians(),
        ))
    };
    let lights: Vec<Arc<dyn Light>> = vec![Arc::new(PointLight {
        emission: Spectrum::one() * 40.0,
        position: Vec3::new(0.3, 4.0, 0.0),
    })];
    let scene = Scene {
        shape,
        camera,
        lights,
    };
    let mut integrator = PathTracer {
        spp: 32,
        max_depth: 5,
    };
    // let mut integrator = SPPM {
    //     initial_radius: 0.1,
    //     iterations: 64,
    //     max_depth: 5,
    //     n_photons: 100000,
    // };
    let film = integrator.render(&scene);
    let image = film.to_rgb_image();
    image.save("out-pt.png").unwrap();
}