import os
from omegaconf import OmegaConf

from dm_control import mjcf
from Pathes import P

JVRC_DESCRIPTION_DIR = os.path.join(P.asset_dir, "jvrc_mj_description")
JVRC_SCENE_XML = os.path.join(JVRC_DESCRIPTION_DIR, "xml/scene.xml")


def builder(export_path, cfg=OmegaConf.load(P.config_dir / "jvrc.yaml")):
    print("Modifying XML model...")
    mjcf_model = mjcf.from_path(JVRC_SCENE_XML)

    mjcf_model.model = cfg.name

    # set njmax and nconmax
    mjcf_model.size.njmax = -1
    mjcf_model.size.nconmax = -1
    mjcf_model.statistic.meansize = 0.1
    mjcf_model.statistic.meanmass = 2

    # modify skybox
    for tx in mjcf_model.asset.texture:
        if tx.type=="skybox":
            tx.rgb1 = '1 1 1'
            tx.rgb2 = '1 1 1'

    # remove all collisions
    mjcf_model.contact.remove()

    # remove actuators except for leg joints
    for mot in mjcf_model.actuator.motor:
        if mot.joint.name not in cfg.joints.leg:
            mot.remove()

    # remove unused joints
    unused_joints_bundle = ["waist", "head", "hand", "arm"]
    unused_joints = []
    for bundle in unused_joints_bundle:
        unused_joints.extend(getattr(cfg.joints, bundle))
    for joint in unused_joints:
        mjcf_model.find('joint', joint).remove()

    # remove existing equality
    mjcf_model.equality.remove()

    # set arm joints to fixed configuration
    for bname, euler in zip(cfg.bodies.arm.name, cfg.bodies.arm.euler):
        mjcf_model.find('body', bname).euler = euler

    # collision geoms
    collision_geoms = [
        'R_HIP_R_S', 'R_HIP_Y_S', 'R_KNEE_S',
        'L_HIP_R_S', 'L_HIP_Y_S', 'L_KNEE_S',
    ]

    # remove unused collision geoms
    for body in mjcf_model.worldbody.find_all('body'):
        for idx, geom in enumerate(body.geom):
            geom.name = body.name + '-geom-' + repr(idx)
            if (geom.dclass.dclass=="collision"):
                if body.name not in collision_geoms:
                    geom.remove()

    # move collision geoms to different group
    mjcf_model.default.default['collision'].geom.group = 3

    # manually create collision geom for feet
    for body in cfg.bodies.foot.name:
        mjcf_model.worldbody.find("body", body).add(
            "geom",
            dclass="collision",
            size=cfg.bodies.foot.collision.size,
            pos=cfg.bodies.foot.collision.pos,
            type=cfg.bodies.foot.collision.type,
        )

    # ignore collision
    mjcf_model.contact.add('exclude', body1='R_KNEE_S', body2='R_ANKLE_P_S')
    mjcf_model.contact.add('exclude', body1='L_KNEE_S', body2='L_ANKLE_P_S')

    # remove unused meshes
    meshes = [g.mesh.name for g in mjcf_model.find_all('geom') if g.type=='mesh' or g.type==None]
    for mesh in mjcf_model.find_all('mesh'):
        if mesh.name not in meshes:
            mesh.remove()

    # fix site pos
    mjcf_model.worldbody.find('site', 'rf_force').pos = '0.03 0.0 -0.1'
    mjcf_model.worldbody.find('site', 'lf_force').pos = '0.03 0.0 -0.1'

    # wrap floor geom in a body
    mjcf_model.find('geom', 'floor').remove()
    mjcf_model.worldbody.add('body', name='floor')
    mjcf_model.find('body', 'floor').add('geom', name='floor', type="plane", size="0 0 0.25", material="groundplane")

    # export model
    mjcf.export_with_assets(mjcf_model, out_dir=export_path, precision=5)
    path_to_xml = os.path.join(export_path, mjcf_model.model + '.xml')
    print("Exporting XML model to ", path_to_xml)
    return


if __name__=='__main__':
    builder(P.tmp_dir / "mjcf-export" / "jvrc_walk")
