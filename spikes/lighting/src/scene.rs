//! A world on the GPU: blocks, mesh and light kept in step through block
//! edits, with the dirty sets an edit needs.

use crate::gpu::Gpu;
use crate::light::{Lighting, UpdateStats};
use crate::mesher::Mesher;
use crate::world::{Rules, World};

pub struct Scene {
    pub world: World,
    pub rules: Rules,
    pub mesher: Mesher,
    pub lighting: Lighting,
}

/// The sections an edit at (x, y, z) from `old` to `new` has to relight.
/// `reset` is zeroed first: light can only fall where the block now blocks
/// more (a placed block) or emits less (a removed torch). Block light
/// reaches 15 blocks, so a block-light fall stays within the 27 sections
/// around; a sky-light fall can run to the bottom of the world under the
/// column, so it takes the 3 x 3 columns of sections from one above down.
/// A rise needs no reset: the section is relaxed and the light spreads.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Dirty {
    pub reset: Vec<u32>,
    pub list: Vec<u32>,
}

pub fn dirty_for(world: &World, rules: &Rules, [x, y, z]: [usize; 3], old: u16, new: u16) -> Dirty {
    let s = world.section_index(x / 16, y / 16, z / 16) as u32;
    let (sx, sy, sz) = ((x / 16) as i32, (y / 16) as i32, (z / 16) as i32);
    let mut reset = Vec::new();
    let mut push = |v: Option<usize>| {
        if let Some(v) = v
            && !reset.contains(&(v as u32))
        {
            reset.push(v as u32);
        }
    };
    if rules.opacity(new) > rules.opacity(old) {
        for dx in -1..=1 {
            for dz in -1..=1 {
                for yy in (0..=sy + 1).rev() {
                    push(world.section_at(sx + dx, yy, sz + dz));
                }
            }
        }
    } else if rules.emission(new) < rules.emission(old) {
        for dx in -1..=1 {
            for dy in -1..=1 {
                for dz in -1..=1 {
                    push(world.section_at(sx + dx, sy + dy, sz + dz));
                }
            }
        }
    }
    let mut list = reset.clone();
    if !list.contains(&s) {
        list.push(s);
    }
    Dirty { reset, list }
}

impl Scene {
    /// Uploads the world, meshes and lights all of it.
    pub fn new(gpu: &Gpu, world: World) -> Scene {
        let rules = Rules::standard();
        let mesher = Mesher::new(gpu, &world, &rules, 1 << 22);
        let lighting = Lighting::new(gpu, &world);
        let scene = Scene {
            world,
            rules,
            mesher,
            lighting,
        };
        scene.mesher.mesh_all(gpu);
        assert_eq!(scene.mesher.overflow(), 0);
        let all: Vec<u32> = (0..scene.world.section_count() as u32).collect();
        scene.lighting.settle(gpu, &scene.mesher, &[], &all);
        scene
    }

    /// Changes one block: uploads its section, remeshes what the change
    /// touches, and relights, a round at a time. Returns the light stats and
    /// the dirty sets used.
    pub fn set_block(&mut self, gpu: &Gpu, p: [usize; 3], new: u16) -> (UpdateStats, Dirty) {
        let [x, y, z] = p;
        let old = self.world.get(x as i32, y as i32, z as i32);
        self.world.set(x, y, z, new);
        let s = self.world.section_index(x / 16, y / 16, z / 16);
        self.mesher.upload_section(&self.world, s);
        let touching = self.world.sections_touching(x, y, z);
        self.mesher.run(gpu, &touching);
        let dirty = dirty_for(&self.world, &self.rules, p, old, new);
        let stats = self
            .lighting
            .settle(gpu, &self.mesher, &dirty.reset, &dirty.list);
        (stats, dirty)
    }

    /// Makes the edit a round at a time to learn its rounds and dirty sets,
    /// undoes it, puts the new block back without relighting, and then
    /// times the light update as one command buffer with the GPU busy.
    /// Leaves the scene holding the edit.
    pub fn time_edit(&mut self, gpu: &Gpu, p: [usize; 3], new: u16) -> (UpdateStats, Dirty, f64) {
        let [x, y, z] = p;
        let old = self.world.get(x as i32, y as i32, z as i32);
        let (stats, dirty) = self.set_block(gpu, p, new);
        self.set_block(gpu, p, old);
        self.world.set(x, y, z, new);
        let s = self.world.section_index(x / 16, y / 16, z / 16);
        self.mesher.upload_section(&self.world, s);
        self.mesher.run(gpu, &self.world.sections_touching(x, y, z));
        let ms =
            self.lighting
                .timed_update(gpu, &self.mesher, &dirty.reset, &dirty.list, stats.rounds);
        (stats, dirty, ms)
    }
}
