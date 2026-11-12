import xml.etree.ElementTree as ET
from init import L1, L2
from Pathes import P

SOURCE_URDF = P.assets_dir / "three_link_arm.urdf"
ADAPTED_URDF = P.assets_dir / "planar_two_link.urdf"


def make_planar_urdf(source, destination):
    """Re-use the uploaded URDF's link identities, with corrected planar joints.

    Source robot has three revolute joints and is NOT a 2R XY robot. Its arm2
    and arm3 become the two links in this task, each scaled to length 10.
    """
    if not source.exists():
        raise FileNotFoundError(f'Missing uploaded URDF: {source}')
    root_in = ET.parse(source).getroot()
    known_links = {link.attrib['name'] for link in root_in.findall('link')}
    if not {'base', 'arm2', 'arm3'} <= known_links:
        raise ValueError(f'Unexpected source links: {known_links}')

    robot = ET.Element('robot', name='two_link_planar_from_three_link_arm')
    ET.SubElement(robot, 'link', name='base')

    for name, length in [('arm2', L1), ('arm3', L2)]:
        link = ET.SubElement(robot, 'link', name=name)
        for tag in ('visual', 'collision'):
            body = ET.SubElement(link, tag)
            ET.SubElement(body, 'origin', xyz=f'{length/2} 0 0.5', rpy='0 0 0')
            geom = ET.SubElement(body, 'geometry')
            ET.SubElement(geom, 'box', size=f'{length} 0.12 0.12')
    for name, parent, child, xyz in [
        ('joint1', 'base', 'arm2', '0 0 0'),
        ('joint2', 'arm2', 'arm3', f'{L1} 0 0'),
    ]:
        j = ET.SubElement(robot, 'joint', name=name, type='revolute')
        ET.SubElement(j, 'parent', link=parent)
        ET.SubElement(j, 'child', link=child)
        ET.SubElement(j, 'origin', xyz=xyz, rpy='0 0 0')
        ET.SubElement(j, 'axis', xyz='0 0 1')
        ET.SubElement(j, 'limit', lower='-6.283185307179586', upper='6.283185307179586', effort='2000', velocity='1000')
    ET.ElementTree(robot).write(destination, encoding='utf-8', xml_declaration=True)


if __name__ == "__main__":
    make_planar_urdf(SOURCE_URDF, ADAPTED_URDF)
