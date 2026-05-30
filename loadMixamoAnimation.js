import * as THREE from 'three';
import { FBXLoader } from 'three/addons/loaders/FBXLoader.js';
import { mixamoVRMRigMap } from './mixamoVRMRigMap.js';

/**
 * Load a Mixamo animation (FBX), retarget it onto a VRM humanoid, and return
 * an AnimationClip whose tracks address the VRM's *normalized* bones.
 *
 * Adapted from the canonical pixiv/three-vrm example, with one addition:
 * a fallback when the FBX clip is not named "mixamo.com" (some Mixamo exports
 * and re-saved FBX files name the clip differently), so arbitrary FBX files
 * still play instead of silently producing nothing.
 *
 * @param {string} url  URL (or object URL) of the Mixamo FBX
 * @param {VRM}    vrm  Target VRM
 * @returns {Promise<THREE.AnimationClip>}
 */
export function loadMixamoAnimation( url, vrm ) {

	const loader = new FBXLoader();
	return loader.loadAsync( url ).then( ( asset ) => {

		// Prefer the standard Mixamo clip name, else fall back to the first clip.
		const clip =
			THREE.AnimationClip.findByName( asset.animations, 'mixamo.com' ) ||
			asset.animations[ 0 ];

		if ( ! clip ) {
			throw new Error( 'No animation clip found in FBX.' );
		}

		const tracks = []; // KeyframeTracks compatible with VRM will be added here

		const restRotationInverse = new THREE.Quaternion();
		const parentRestWorldRotation = new THREE.Quaternion();
		const _quatA = new THREE.Quaternion();

		// Detect the rig's bone-name prefix. Mixamo is usually "mixamorig" but
		// re-rigged/re-uploaded characters get numbered prefixes ("mixamorig1",
		// "mixamorig2", ...); FBXLoader also strips any ":" from "mixamorig:Hips".
		// Find the hips bone whatever it's called and derive the prefix from it.
		let hipsNode = null;
		asset.traverse( ( o ) => {
			if ( ! hipsNode && /mixamorig/i.test( o.name ) && /Hips$/i.test( o.name ) ) hipsNode = o;
		} );
		if ( ! hipsNode ) {
			throw new Error( 'No Mixamo "Hips" bone found. Is this a Mixamo animation FBX (mixamorig skeleton), not a character mesh?' );
		}
		const rigPrefix = hipsNode.name.replace( /Hips$/i, '' ); // e.g. "mixamorig" or "mixamorig1"
		// Map an actual FBX bone name to the canonical "mixamorig*" key used by the rig map.
		const toCanonical = ( name ) => 'mixamorig' + name.slice( rigPrefix.length );

		// Adjust with reference to hips height.
		const motionHipsHeight = hipsNode.position.y;
		const vrmHipsHeight = vrm.humanoid.normalizedRestPose.hips.position[ 1 ];
		const hipsPositionScale = vrmHipsHeight / motionHipsHeight;

		clip.tracks.forEach( ( track ) => {

			const trackSplitted = track.name.split( '.' );
			const mixamoRigName = trackSplitted[ 0 ];
			const vrmBoneName = mixamoVRMRigMap[ toCanonical( mixamoRigName ) ];
			const vrmNodeName = vrm.humanoid?.getNormalizedBoneNode( vrmBoneName )?.name;
			const mixamoRigNode = asset.getObjectByName( mixamoRigName );

			if ( vrmNodeName != null ) {

				const propertyName = trackSplitted[ 1 ];

				// Store rotations of rest-pose.
				mixamoRigNode.getWorldQuaternion( restRotationInverse ).invert();
				mixamoRigNode.parent.getWorldQuaternion( parentRestWorldRotation );

				if ( track instanceof THREE.QuaternionKeyframeTrack ) {

					// Retarget rotation of mixamoRig to NormalizedBone.
					for ( let i = 0; i < track.values.length; i += 4 ) {

						const flatQuaternion = track.values.slice( i, i + 4 );

						_quatA.fromArray( flatQuaternion );

						// parentRestWorldRotation * trackRotation * restWorldRotationInverse
						_quatA
							.premultiply( parentRestWorldRotation )
							.multiply( restRotationInverse );

						_quatA.toArray( flatQuaternion );

						flatQuaternion.forEach( ( v, index ) => {

							track.values[ index + i ] = v;

						} );

					}

					tracks.push(
						new THREE.QuaternionKeyframeTrack(
							`${vrmNodeName}.${propertyName}`,
							track.times,
							track.values.map( ( v, i ) => ( vrm.meta?.metaVersion === '0' && i % 2 === 0 ? - v : v ) ),
						),
					);

				} else if ( track instanceof THREE.VectorKeyframeTrack ) {

					const value = track.values.map( ( v, i ) => ( vrm.meta?.metaVersion === '0' && i % 3 !== 1 ? - v : v ) * hipsPositionScale );
					tracks.push( new THREE.VectorKeyframeTrack( `${vrmNodeName}.${propertyName}`, track.times, value ) );

				}

			}

		} );

		return new THREE.AnimationClip( 'vrmAnimation', clip.duration, tracks );

	} );

}
