import xatlas
import trimesh
import argparse

# def parameterize_mesh(input_path, output_path):
#     # parameterize the mesh
#     mesh = trimesh.load_mesh(input_path, force='mesh')
#     vertices, faces = mesh.vertices, mesh.faces

#     vmapping, indices, uvs = xatlas.parametrize(vertices, faces)
#     xatlas.export(str(output_path), vertices[vmapping], indices, uvs)

def parameterize_mesh(input_path, output_path):
    # Load the mesh
    mesh = trimesh.load_mesh(input_path, force='mesh')
    
    # Ensure the mesh has vertex normals
    if not hasattr(mesh, 'vertex_normals') or mesh.vertex_normals is None:
        mesh.vertex_normals = mesh.vertex_normals  # Automatically computes if not present

    vertices, faces = mesh.vertices, mesh.faces
    normals = mesh.vertex_normals  # Vertex normals
    
    # Parameterize the mesh using xatlas
    vmapping, indices, uvs = xatlas.parametrize(vertices, faces)
    
    # Export the mesh with the new UVs and ensure normals are included
    xatlas.export(
        str(output_path),
        vertices[vmapping], 
        indices, 
        uvs,
        normals=normals[vmapping]  # Pass vertex normals if required by export format
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    args = parser.parse_args()

    parameterize_mesh(args.input_path, args.output_path)
    