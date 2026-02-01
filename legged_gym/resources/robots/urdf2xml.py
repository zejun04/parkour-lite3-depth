#!/usr/bin/env python3
"""
URDF to MuJoCo XML converter for Lite3 robot
Converts URDF file to MuJooco-compatible XML format
"""

import xml.etree.ElementTree as ET
import os


def parse_urdf_element(element):
    """Parse URDF element and return as string"""
    return ET.tostring(element, encoding='unicode')


def convert_urdf_to_mujoco(urdf_path, output_path, mesh_dir=None):
    """
    Convert URDF file to MuJoCo XML format
    
    Args:
        urdf_path: Path to input URDF file
        output_path: Path to output MuJoCo XML file
        mesh_dir: Directory containing mesh files (relative to XML)
    """
    
    # Parse URDF file
    tree = ET.parse(urdf_path)
    root = tree.getroot()
    
    # Create MuJoCo XML structure
    mujoco = ET.Element("mujoco")
    mujoco.set("model", "lite3")
    
    # Compiler section
    compiler = ET.SubElement(mujoco, "compiler")
    compiler.set("angle", "radian")
    compiler.set("meshdir", mesh_dir if mesh_dir else "meshes")
    compiler.set("balancedmemory", "true")
    
    # Default section for joints and geoms
    default = ET.SubElement(mujoco, "default")
    
    # Default joint settings
    joint_class = ET.SubElement(default, "joint")
    joint_class.set("limited", "true")
    joint_class.set("damping", "0.1")
    joint_class.set("frictionloss", "0.01")
    
    # Default geom settings
    geom_class = ET.SubElement(default, "geom")
    geom_class.set("margin", "0.001")
    geom_class.set("condim", "3")
    geom_class.set("friction", "0.5 0.005 0.0001")
    geom_class.set("contype", "1")
    geom_class.set("conaffinity", "1")
    
    # Worldbody section
    worldbody = ET.SubElement(mujoco, "worldbody")
    
    # Map to store link positions (for joint transformations)
    link_positions = {}
    
    # Process all links and joints
    links = root.findall("link")
    joints = root.findall("joint")
    
    # Build a map of parent-child relationships
    joint_map = {}
    for joint in joints:
        parent = joint.find("parent")
        child = joint.find("child")
        if parent is not None and child is not None:
            joint_map[child.get("link")] = {
                'joint': joint,
                'parent': parent.get("link")
            }
    
    # Find root link (no parent)
    root_link = None
    for link in links:
        if link.get("name") not in joint_map:
            root_link = link
            break
    
    if root_link is None:
        root_link = links[0]  # Fallback to first link
    
    # Process links recursively
    processed_links = set()
    
    def process_link(link_name, parent_body=None):
        if link_name in processed_links:
            return
        processed_links.add(link_name)
        
        # Find the link element
        link_elem = None
        for link in links:
            if link.get("name") == link_name:
                link_elem = link
                break
        
        if link_elem is None:
            return
        
        # Create body for this link
        if parent_body is None:
            # Root body
            body = ET.SubElement(worldbody, "body")
            body.set("name", link_name)
            body.set("pos", "0 0 0")
        else:
            # Child body - get joint info
            joint_info = joint_map.get(link_name)
            if joint_info is None:
                return
            
            joint = joint_info['joint']
            origin = joint.find("origin")
            axis = joint.find("axis")
            
            pos = "0 0 0"
            if origin is not None:
                pos = origin.get("xyz", "0 0 0").replace(",", " ")
            
            body = ET.SubElement(parent_body, "body")
            body.set("name", link_name)
            body.set("pos", pos)
        
        # Add inertial properties
        inertial = link_elem.find("inertial")
        if inertial is not None:
            origin = inertial.find("origin")
            mass = inertial.find("mass")
            inertia = inertial.find("inertia")
            
            if mass is not None:
                body.set("mass", mass.get("value"))
            
            if inertia is not None:
                # MuJoCo uses diag for diagonal inertia
                body.set("inertiagroup", "1")
        
        # Add visual/collision geoms
        visuals = link_elem.findall("visual")
        collisions = link_elem.findall("collision")
        
        # Process both visual and collision
        for is_collision in [False, True]:
            elements = collisions if is_collision else visuals
            
            for elem in elements:
                origin = elem.find("origin")
                geometry = elem.find("geometry")
                
                if geometry is None:
                    continue
                
                geom = ET.SubElement(body, "geom")
                geom.set("type", "none") if not is_collision else None
                
                # Get position and orientation
                if origin is not None:
                    pos = origin.get("xyz", "0 0 0").replace(",", " ")
                    geom.set("pos", pos)
                    
                    euler = origin.get("rpy", "0 0 0").replace(",", " ")
                    if euler != "0 0 0":
                        geom.set("euler", euler)
                
                # Process geometry
                mesh_elem = geometry.find("mesh")
                box_elem = geometry.find("box")
                sphere_elem = geometry.find("sphere")
                cylinder_elem = geometry.find("cylinder")
                
                if mesh_elem is not None:
                    filename = mesh_elem.get("filename", "")
                    if "../meshes/" in filename:
                        filename = filename.replace("../meshes/", "")
                    geom.set("mesh", filename)
                    geom.set("type", "mesh")
                elif box_elem is not None:
                    size = box_elem.get("size", "0.1 0.1 0.1").replace(",", " ")
                    geom.set("size", size)
                    geom.set("type", "box")
                elif sphere_elem is not None:
                    radius = sphere_elem.get("radius", "0.05")
                    geom.set("size", radius)
                    geom.set("type", "sphere")
                elif cylinder_elem is not None:
                    radius = cylinder_elem.get("radius", "0.05")
                    length = cylinder_elem.get("length", "0.1")
                    geom.set("size", f"{radius} {length/2}")
                    geom.set("type", "cylinder")
                
                if not is_collision:
                    geom.set("contype", "0")  # Visual geoms don't collide
                    geom.set("conaffinity", "0")
                else:
                    geom.set("contype", "1")
                    geom.set("conaffinity", "1")
        
        # Add joint (if this link is a child)
        joint_info = joint_map.get(link_name)
        if joint_info is not None:
            joint = joint_info['joint']
            joint_type = joint.get("type", "fixed")
            
            if joint_type == "revolute":
                mj_joint = ET.SubElement(body, "joint")
                mj_joint.set("name", joint.get("name"))
                mj_joint.set("type", "hinge")
                
                # Axis
                axis = joint.find("axis")
                if axis is not None:
                    axis_val = axis.get("xyz", "0 0 1").replace(",", " ")
                    mj_joint.set("axis", axis_val)
                
                # Limits
                limit = joint.find("limit")
                if limit is not None:
                    lower = limit.get("lower", "-3.14")
                    upper = limit.get("upper", "3.14")
                    mj_joint.set("range", f"{lower} {upper}")
                    
                    effort = limit.get("effort", "30")
                    mj_joint.set("actuatorfrcrange", f"-{effort} {effort}")
        
        # Find and process children
        for joint_name, info in joint_map.items():
            if info['parent'] == link_name:
                process_link(joint_name, body)
    
    # Start processing from root link
    process_link(root_link.get("name"))
    
    # Add actuator section
    actuator = ET.SubElement(mujoco, "actuator")
    
    # Add motors for revolute joints
    for joint in joints:
        if joint.get("type") == "revolute":
            motor = ET.SubElement(actuator, "motor")
            motor.set("name", joint.get("name") + "_motor")
            motor.set("joint", joint.get("name"))
            motor.set("gear", "1.0")
            
            limit = joint.find("limit")
            if limit is not None:
                effort = limit.get("effort", "30")
                motor.set("forcerange", f"-{effort} {effort}")
    
    # Write to file with proper formatting
    def prettify(elem):
        """Return a pretty-printed XML string"""
        rough_string = ET.tostring(elem, encoding='unicode')
        from xml.dom import minidom
        reparsed = minidom.parseString(rough_string)
        return reparsed.toprettyxml(indent="  ")
    
    # Create directory if needed
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Write the file
    with open(output_path, 'w') as f:
        f.write('<?xml version="1.0" ?>\n')
        f.write('<mujoco model="lite3">\n')
        
        # Write compiler
        f.write('  <compiler angle="radian" meshdir="../resources/robots/lite3/meshes"/>\n\n')
        
        # Write defaults
        f.write('  <default>\n')
        f.write('    <joint limited="true" damping="0.1" frictionloss="0.01"/>\n')
        f.write('    <geom margin="0.001" condim="3" friction="0.5 0.005 0.0001" contype="1" conaffinity="1"/>\n')
        f.write('  </default>\n\n')
        
        # Write worldbody
        f.write('  <worldbody>\n')
        
        # Process links manually for better control
        write_links_to_xml(f, root, links, joints)
        
        f.write('  </worldbody>\n\n')
        
        # Write actuators
        f.write('  <actuator>\n')
        write_actuators_to_xml(f, joints)
        f.write('  </actuator>\n')
        f.write('</mujoco>\n')
    
    print(f"Successfully converted {urdf_path} to {output_path}")


def write_links_to_xml(f, root, links, joints):
    """Write links to XML file with proper formatting"""
    
    # Build joint map
    joint_map = {}
    for joint in joints:
        parent = joint.find("parent")
        child = joint.find("child")
        if parent is not None and child is not None:
            joint_map[child.get("link")] = {
                'joint': joint,
                'parent': parent.get("link")
            }
    
    # Find root link
    root_link = None
    for link in links:
        if link.get("name") not in joint_map:
            root_link = link
            break
    
    if root_link is None:
        root_link = links[0]
    
    # Process links recursively
    processed = set()
    
    def write_link(link_name, parent_name=None, indent="    "):
        if link_name in processed:
            return
        processed.add(link_name)
        
        link = None
        for l in links:
            if l.get("name") == link_name:
                link = l
                break
        
        if link is None:
            return
        
        # Get position from joint if not root
        pos = "0 0 0"
        euler = "0 0 0"
        
        joint_info = joint_map.get(link_name)
        if joint_info is not None and parent_name:
            origin = joint_info['joint'].find("origin")
            if origin is not None:
                pos = origin.get("xyz", "0 0 0").replace(",", " ")
                euler = origin.get("rpy", "0 0 0").replace(",", " ")
        
        # Write body tag
        f.write(f'{indent}<body name="{link_name}" pos="{pos}" euler="{euler}">\n')
        
        # Write inertial
        inertial = link.find("inertial")
        if inertial is not None:
            mass = inertial.find("mass")
            inertia = inertial.find("inertia")
            origin = inertial.find("origin")
            if mass is not None:
                pos_attr = ' pos="0 0 0"'
                if origin is not None:
                    pos_val = origin.get("xyz", "0 0 0").replace(",", " ")
                    pos_attr = f' pos="{pos_val}"'
                
                # Add diagonal inertia if available
                diaginertia_attr = ""
                if inertia is not None:
                    ixx = inertia.get("ixx", "0.001")
                    iyy = inertia.get("iyy", "0.001")
                    izz = inertia.get("izz", "0.001")
                    diaginertia_attr = f' diaginertia="{ixx} {iyy} {izz}"'
                
                f.write(f'{indent}  <inertial mass="{mass.get("value")}"{pos_attr}{diaginertia_attr}/>\n')
        else:
            # Default inertial for links without one (required by MuJoCo)
            f.write(f'{indent}  <inertial mass="0.001" pos="0 0 0" diaginertia="1e-6 1e-6 1e-6"/>\n')
        
        # Write geoms (collision and visual)
        write_geoms(f, link, indent + "  ")
        
        # Write joint if revolute
        if joint_info is not None:
            joint = joint_info['joint']
            if joint.get("type") == "revolute":
                axis = joint.find("axis")
                limit = joint.find("limit")
                
                axis_val = "0 0 1"
                if axis is not None:
                    axis_val = axis.get("xyz", "0 0 1").replace(",", " ")
                
                range_val = "-3.14 3.14"
                if limit is not None:
                    lower = limit.get("lower", "-3.14")
                    upper = limit.get("upper", "3.14")
                    range_val = f"{lower} {upper}"
                
                f.write(f'{indent}  <joint name="{joint.get("name")}" type="hinge" axis="{axis_val}" range="{range_val}"/>\n')
        
        # Find and write children
        for child_name, info in joint_map.items():
            if info['parent'] == link_name:
                write_link(child_name, link_name, indent + "  ")
        
        f.write(f'{indent}</body>\n')
    
    write_link(root_link.get("name"))


def write_geoms(f, link, indent):
    """Write geometry elements for a link"""
    
    # Process collision geoms first (for physics)
    collisions = link.findall("collision")
    for collision in collisions:
        write_geom(f, collision, indent, is_collision=True)
    
    # Process visual geoms (for rendering)
    visuals = link.findall("visual")
    for visual in visuals:
        write_geom(f, visual, indent, is_collision=False)


def write_geom(f, elem, indent, is_collision):
    """Write a single geometry element"""
    origin = elem.find("origin")
    geometry = elem.find("geometry")
    
    if geometry is None:
        return
    
    pos = "0 0 0"
    euler = "0 0 0"
    if origin is not None:
        pos = origin.get("xyz", "0 0 0").replace(",", " ")
        euler = origin.get("rpy", "0 0 0").replace(",", " ")
    
    geom_type = "none"
    size_info = ""
    mesh_name = ""
    
    mesh_elem = geometry.find("mesh")
    box_elem = geometry.find("box")
    sphere_elem = geometry.find("sphere")
    cylinder_elem = geometry.find("cylinder")
    
    if mesh_elem is not None:
        geom_type = "mesh"
        filename = mesh_elem.get("filename", "")
        if "../meshes/" in filename:
            mesh_name = filename.replace("../meshes/", "")
    elif box_elem is not None:
        geom_type = "box"
        size = box_elem.get("size", "0.1 0.1 0.1").replace(",", " ")
        size_info = f' size="{size}"'
    elif sphere_elem is not None:
        geom_type = "sphere"
        radius = sphere_elem.get("radius", "0.05")
        size_info = f' size="{radius}"'
    elif cylinder_elem is not None:
        geom_type = "cylinder"
        radius = cylinder_elem.get("radius", "0.05")
        length = cylinder_elem.get("length", "0.1")
        size_info = f' size="{radius} {length/2}"'
    
    if is_collision:
        contype = "1"
        conaffinity = "1"
    else:
        contype = "0"
        conaffinity = "0"
    
    if geom_type == "mesh":
        f.write(f'{indent}<geom type="{geom_type}" mesh="{mesh_name}" pos="{pos}" euler="{euler}" contype="{contype}" conaffinity="{conaffinity}"/>\n')
    else:
        f.write(f'{indent}<geom type="{geom_type}" pos="{pos}" euler="{euler}" contype="{contype}" conaffinity="{conaffinity}"{size_info}/>\n')


def write_actuators_to_xml(f, joints):
    """Write actuator section"""
    indent = "    "
    for joint in joints:
        if joint.get("type") == "revolute":
            limit = joint.find("limit")
            effort = "30"
            if limit is not None:
                effort = limit.get("effort", "30")
            
            f.write(f'{indent}<motor name="{joint.get("name")}_motor" joint="{joint.get("name")}" gear="1.0" forcerange="-{effort} {effort}"/>\n')


if __name__ == "__main__":
    # Paths
    urdf_file = "/home/shenlan/RL_gym/extreme-parkour/legged_gym/resources/robots/lite3/urdf/Lite3.urdf"
    output_file = "/home/shenlan/RL_gym/extreme-parkour/legged_gym/resources/robots/lite3/urdf/Lite3.xml"
    # meshdir should be relative to deploy/ directory
    mesh_directory = "../resources/robots/lite3/meshes"
    
    # Convert
    convert_urdf_to_mujoco(urdf_file, output_file, mesh_directory)
    
    print("\nConversion complete!")
    print(f"Output file: {output_file}")
