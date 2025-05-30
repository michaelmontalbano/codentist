import React, { Suspense, useRef } from 'react'
import { Canvas } from '@react-three/fiber'
import { OrbitControls, useGLTF, OrthographicCamera } from '@react-three/drei'
import * as THREE from 'three'

const toothConfig = [
  { name: 'maxillary_third_molar', position: [-12, -3, 0], isRightSide: true }, // 0
  { name: 'maxillary_second_molar', position: [-7.7, -2.5, -1.6], isRightSide: true }, // 1
  { name: 'maxillary_first_molar', position: [-8, 0.2, 1], isRightSide: true }, // 2
  { name: 'maxillary_second_premolar', position: [-7, 0, 0], isRightSide: true }, // 3
  { name: 'maxillary_first_premolar', position: [-6, 0.2, -0.3], isRightSide: true }, // 4
  { name: 'maxillary_canine', position: [-5, 0.6, -.5], isRightSide: true },         // 5
  { name: 'maxillary_lateral_incisor', position: [-4.5, 0.2, -1.2], isRightSide: true }, // 6
  { name: 'maxillary_left_central_incisor', position: [-4, 0.4, -.6], isRightSide: true },  // 7
  { name: 'maxillary_left_central_incisor', position: [-3.3, 0.4, -.6], isRightSide: false }, // 8
  { name: 'maxillary_lateral_incisor', position: [-2.8, 0.2, -1.3], isRightSide: false }, // 9
  { name: 'maxillary_canine', position: [-2.2, 0.6, -.7], isRightSide: false }, // 10
  { name: 'maxillary_first_premolar', position: [-1.5, 0.2, -.5], isRightSide: false }, // 11
  { name: 'maxillary_second_premolar', position: [-.8, 0, -.35], isRightSide: false }, // 12
  { name: 'maxillary_first_molar', position: [.3, 0.2, .6], isRightSide: false }, // 13
  { name: 'maxillary_second_molar', position: [0, -2.5, -1.6], isRightSide: false },
  { name: 'maxillary_third_molar', position: [7.5, 0, 0], isRightSide: false }
]

function Tooth({ name, position, rotation, isRightSide = false }) {
  const path = `teeth3D/${name}.glb`
  const isMolar = name.includes('third_molar') || name.includes('second_molar')
  console.log(`Attempting to load: ${path}`)

  try {
    const { scene } = useGLTF(path)
    if (!scene) {
      console.warn(`Model failed to load: ${path}`)
      return (
        <mesh position={position} rotation={rotation} scale={[0.5, 0.5, 0.5]}>
          <boxGeometry args={[1, 1, 1]} />
          <meshStandardMaterial color="red" />
        </mesh>
      )
    }

    const cloned = scene.clone(true)

    cloned.traverse((child) => {
      if (child.isMesh) {
        if (child.material) {
          const originalMaterial = child.material
          child.material = originalMaterial.clone()
          child.material.metalness = Math.min((child.material.metalness || 0) + 0.1, 1)
          child.material.roughness = Math.max((child.material.roughness || 0.5) - 0.3, 0.1)

          if (child.material.isMeshPhysicalMaterial || child.material.isMeshStandardMaterial) {
            child.material.clearcoat = 0.5
            child.material.clearcoatRoughness = 0.1
          }
        }
        child.castShadow = true
        child.receiveShadow = true
      }
    })

    if (isRightSide) {
      cloned.scale.set(-1.2, 1.2, 1.2) // Mirror on X-axis
    } else {
      cloned.scale.set(1.2, 1.2, 1.2)
    }

    return <primitive object={cloned} position={position} rotation={rotation} />
  } catch (e) {
    console.error(`Error loading ${path}:`, e)
    return (
      <mesh position={position} rotation={rotation} scale={[0.5, 0.5, 0.5]}>
        <boxGeometry args={[1, 1, 1]} />
        <meshStandardMaterial color="orange" />
      </mesh>
    )
  }
}

export default function ToothChart3D() {
  const groupRef = useRef()

  return (
    <Canvas style={{ height: '100vh', width: '100vw' }} shadows>
      <OrthographicCamera 
        makeDefault
        position={[0, 5, 25]} 
        zoom={20}
        near={0.1}
        far={1000}
      />
      <ambientLight intensity={0.6} />
      <directionalLight 
        position={[10, 10, 5]} 
        intensity={1.0} 
        castShadow
        shadow-mapSize-width={2048}
        shadow-mapSize-height={2048}
      />
      <directionalLight position={[-5, 8, 5]} intensity={0.8} />
      <pointLight position={[0, 10, 0]} intensity={0.5} />

      <Suspense fallback={null}>
        <group ref={groupRef}>
          {toothConfig.map((tooth, i) => (
            <Tooth
              key={`tooth_${i}`}
              name={tooth.name}
              position={tooth.position}
              rotation={[0, 0, 0]}
              isRightSide={tooth.isRightSide}
            />
          ))}
        </group>
      </Suspense>

      <OrbitControls 
        enablePan={true} 
        enableZoom={true} 
        enableRotate={true}
        minDistance={0.01}
        maxDistance={500}
        maxPolarAngle={Math.PI}
        minPolarAngle={0}
        panSpeed={3}
        zoomSpeed={5}
        rotateSpeed={1}
        enableDamping={false}
      />
    </Canvas>
  )
}
