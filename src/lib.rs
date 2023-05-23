/*Copyright (c) 2022 Diego da Silva Lima. All rights reserved.

This work is licensed under the terms of the MIT license.  
For a copy, see <https://opensource.org/licenses/MIT>.*/

use std::error::Error;
use std::fmt::{Debug};
use std::rc::Rc;
use std::cell::RefCell;
use std::boxed;
use std::collections::{BTreeMap, HashMap, VecDeque};

/**
This is a minimal design pattern crate exposing the following:

![Typestate pattern](http://cliffle.com/blog/rust-typestate/): Transition trait.
Java-style OO inheritance: Inherit trait.
Modula-style late-binding/message-passing/signal-slot: Traits React and Message; types Callbacks<.> and ValuedCallbacks<.>, which
are containers for slots.

The crate uses only abstractions from the standard library and brings no transitive dependencies.

Its main motivation was centralizing a set of traits that I kept re-inventing in my crates. As such, the crate provides
minimal functionality, it just expose maximally generic types and traits useful for GUI code, to improve the software
engineering side of things. Hopefully the examples are enough to represent how they might be useful for GUI code. I
use the crate extensively to organize applications written in GTK, but the abstractions should be agnostic to GUI framework.
**/

pub trait PersistentState<T> : Sized {

    /// Attempts to open UserState by deserializing it from a JSON path.
    /// This is a blocking operation.
    fn recover(path : &str) -> Option<Self>;

    fn update(&self, target : &T);

    // Saves the state to the given path by spawning a thread. This is
    // a nonblocking operation.
    fn persist(&self, path : &str) -> std::thread::JoinHandle<bool>;

}

/// Like Callbacks<A>, but defines both a set of arguments and a return type.
#[derive(Clone, Default)]
pub struct ValuedCallbacks<A, R>(Rc<RefCell<Vec<boxed::Box<dyn Fn(A)->R + 'static>>>>);

pub trait Message {

}

/// Holds a set of callbacks with a given signature. Useful if several objects must respond to
/// the same signal. If more than one object must be taken as argument, use a custom struct or
/// tuples.
pub type Callbacks<A> = ValuedCallbacks<A, ()>;

impl<A, R> ValuedCallbacks<A, R> {

    pub fn new() -> Self {
        Self(Rc::new(RefCell::new(Vec::new())))
    }

}

// pub type Callbacks<A> = Rc<RefCell<Vec<boxed::Box<dyn Fn(A) + 'static>>>>;

pub type BindResult<T> = Result<T, Box<dyn Error>>;

impl<A, R> ValuedCallbacks<A, R> {

    pub fn try_call(&self, args : A) -> BindResult<()>
    where
        A :  Clone
    {
        self.0.try_borrow()?.iter().for_each(|f| { f(args.clone()); } );
        Ok(())
    }

    pub fn call(&self, args : A)
    where
        A :  Clone
    {
        self.0.borrow().iter().for_each(|f| { f(args.clone()); });
    }

    pub fn try_call_with_values(&self, args : A) -> BindResult<Vec<R>>
    where
        A :  Clone,
        R : Sized
    {
        Ok(self.0.try_borrow()?.iter().map(|f| f(args.clone()) ).collect())
    }

    pub fn call_with_values(&self, args : A) -> Vec<R>
    where
        A :  Clone,
        R : Sized
    {
        self.0.borrow().iter().map(|f| f(args.clone()) ).collect()
    }

    pub fn try_bind(&self, f : impl Fn(A)->R + 'static) -> BindResult<()> {
        self.0.try_borrow_mut()?.push(boxed::Box::new(f));
        Ok(())
    }

    pub fn bind(&self, f : impl Fn(A)->R + 'static) {
        self.0.borrow_mut().push(boxed::Box::new(f));
    }

    pub fn try_count_bounded(&self) -> BindResult<usize> {
        Ok(self.0.try_borrow()?.len())
    }

    pub fn count_bounded(&self) -> usize {
        self.0.borrow().len()
    }

}

/// Generic trait to represent interactions between Views (widgets or sets of grouped widgets affected by data change),
/// Models (data structures that encapsulate data-modifying algorithms) and controls (widgets
/// that modify the data contained in models). Widgets that affect a model (Controls) are represented by having the model imlement React<Widget>.
/// Widgets that are affected by a model (Views) are represented by having the widget implement React<Model>.
/// The implementation will usually bind one or more closures to the argument. Usually, an widget is either a control OR a view
/// with respect to a given model, but might a assume both roles. A widget might also be a view for one model but the control for another model. Models
/// usually encapsulate a call to glib::Receiver::attach(.), waiting for messages that change their state. The models are implemented
/// by closures activated on "signals", implemented using Rust enums. The actual data is not held in the model structure, but is owned
/// by a single closure executing on the main thread whenever some signal enum is received. If required, the model might spawn new
/// threads or wait for response from worker threads, but they should never block.
pub trait React<S> {

    fn react(&self, source : &S);

}

/// Stateful algorithms have any computations potentially depending on
/// their state. This trait simply makes this state explicit on a single
/// type, rather than dispersed across fields in the structure. Tipically,
/// state can be an enum or struct for a type T named TState.
pub trait Stateful {

    type State;

    fn state(&self) -> &Self::State;

    // pub trait Clear where Self : Stateful { fn clear(&mut self); }

}

// Perhaps add a derive macro for stateless: Its derivation can only succeed if
// no field implements Stateful. Then the user must implement stateful for
// all algorithms whose computation depend on the internal state. Stateless
// is useful to denote structures that compute in/out via &mut self, but
// use the inner state only as a cache and to avoid reallocations.
pub trait Stateless {

}

/// Attempts to initialize an object.
pub trait Initialize
where
    Self : Stateful + Sized
{

    // fn initialize(&mut self) -> Result<Self, <Self as Stateful>::Error>;

}

/// Attempts to perform an operation after which the object will not be used anymore.
/// Taking the object by value guarantees it won't be used anymore.
pub trait Finalize
where
    Self : Stateful
{

    // fn finalize(self) -> Result<(), <Self as Stateful>::Error>;

}

/// TODO move to verifiable crate, and make stateful dependent on it.
/// Trait implemented by stateful structs or enumerations for which
/// certain invariants must be held and can be verified at runtime.
/// You can #[derive(Verify)] if all fields of your structure satisfy
/// verify, in which case the error will be Box<dyn Error> wrapping the
/// first violation found.
pub trait Verify
where
    Self : Debug + Sized
{

    type Error : std::error::Error;

    fn verify(&self) -> Result<(), Self::Error>;

    /// # Panic
    /// Panics when the structure is not in a valid state.
    fn assume_verified(&self) {
        match self.verify() {
            Err(e) => panic!("Invariant violated for {:?}: {}", self, e),
            _ => { },
        }
    }

    fn is_verified(&self) -> bool {
        self.verify().is_ok()
    }

    /// Applies the given closure, returning the object back to the user
    /// if it is at a valid state after the closure is applied to it, and
    /// consuming it if the closure left it at an invalid state.
    ///
    /// The example assume that at the default state the object is valid.
    /// # Example
    ///
    /// use std::mem;
    /// match value.apply(mem::take(&mut obj), |obj| *obj += 1 ) {
    ///    Ok(obj) => { *obj = obj },
    ///    Err(e) => { }
    /// }
    fn apply(mut self, mut f : impl FnMut(&mut Self)) -> Result<Self, Self::Error> {
        f(&mut self);
        match self.verify() {
            Ok(_) => Ok(self),
            Err(e) => Err(e)
        }
    }

}

/*pub struct SliceError<T, E> {
    ix : usize,
    err : E
}

impl<E> Verify for [T]
where
    T : Verify<Error=E>
{
    fn verify(&self) -> Result<(), SliceError<E> {
        self.iter().map(|v| v.verify() )
    }
}*/

/*/// A common trait for state machines.
/// Trait implemented by structs whose state can be characterized as having one of a few
/// discrete states at any given time. stateful structs or enumerations (usually enumerations)
/// that can be at one of a few states known at compile time. Some of the transitions
/// from state T->U might be invalid and must be specified at the implementation
/// of self.transition. TODO rename to transition?
pub trait Transition
where
    Self : Debug + Stateful + Sized
{

    type TError : std::error::Error;

    /// Calls Self::try_transition, panicking when the transition is not valid.
    fn transition(&mut self, to : Self) {
        // It would be nice to show the destination state, but we take to by value instead of by ref.
        match self.try_transition(to) {
            Err(e) => panic!("Invalid transition from {:?}: {}", self, e),
            _ => { }
        }
    }

    /// Modifies Self by setting it to another value at to. It might be the case
    /// that there isn't a valid transition between the current state and the state
    /// represented at to, in which case
    fn try_transition(&mut self, to : Self) -> Result<(), Self::TError>;

    /// Checks if a transition is legal by verifying the current state and
    /// the next possible state before attempting the transition.
    /// If transition from state T->U is invalid, overwrite this method.
    fn verified_transition(
        &mut self,
        to : Self
    ) -> Result<(), Self::TError>
    where
        Self : Verify
    {
        self.verify()?;
        to.verify()?;
        self.transition(to);
        Ok(())
    }

}*/

/** Trait useful for Java-style inheritance. It allows T to implement 
all of parent's methods if the parent's methods live in a trait with 
a default implementation for T : Inherit<Self>. Note this trait only allows
for single-parent relationships (tree-like inheritance diagram). For multiple
inheritance (DAG-like inheritance diagram) Parent should be a trait type argument.
A way to simulate multiple inheritance is to use Type Parent = (A, B), then to use
the fields/methods of A use parent().0 and to use the fields/methods of B use parent().1 **/
pub trait Inherit {
    
    type Parent;
    
    // type ParentImpl;
    
    /* Returns a reference to this type's parent */
    fn parent<'a>(&'a self) -> &'a Self::Parent;
    
    /* Returns a mutable reference to this type's parent */
    fn parent_mut<'a>(&'a mut self) -> &'a mut Self::Parent;
    
}

/*/// Assumes an operation is executed within the specified time. Perhaps
/// Best implemented via a macro to be applied over the function
/// call.
pub trait Timed {

    fn timed()

}*/

pub trait Take<'a, T>
where
    T : Default
{

    fn take(self) -> Taken<'a, T>;

}

pub struct Taken<'a, T>
where
    T : Default
{
    loc : &'a mut T,
    val : T
}

impl<'a, T> Taken<'a, T>
where
    T : Default
{
    // After this step, the drop impl will replace the original
    // with default, leaving the T free for the user.
    pub fn steal(mut self) -> T {
        std::mem::take(&mut self.val)
    }
}

impl<'a, T> Drop for Taken<'a, T>
where
    T : Default
{

    fn drop(&mut self) {
        *self.loc = std::mem::take(&mut self.val);
    }

}

impl<'a, T> Take<'a, T> for &'a mut T
where
    Self : Default,
    T : Default
{

    fn take(self) -> Taken<'a, T> {
        Taken { val : std::mem::take(self), loc : self }
    }

}

impl<'a, T> std::ops::Deref for Taken<'a, T>
where
    T : Default
{

    type Target = T;

    fn deref(&self) -> &T {
        &self.val
    }
}

impl<'a, T> std::ops::DerefMut for Taken<'a, T>
where
    T : Default
{

    fn deref_mut(&mut self) -> &mut T {
        &mut self.val
    }

}

#[derive(Clone, Default)]
pub struct S(Vec<i32>);

use std::thread::LocalKey;
use std::ops::{Deref, DerefMut};

// The Default is required for the Drop implementation of Unique<T>,
// since this moves the value out of the Unique<T> back into the
// singleton thread-local storage.
pub trait Singleton
where
    Self : Clone + Default + 'static
{

    fn instance() -> &'static LocalKey<RefCell<Option<Self>>>;

    fn view<F>(mut f : F)
    where
        F : FnMut(&Self)
    {
        Self::instance().with(|s| f(s.borrow().as_ref().unwrap()) );
    }

    fn update<F>(mut f : F)
    where
        F : FnMut(&mut Self)
    {
        Self::instance().with(|s| f(s.borrow_mut().as_mut().unwrap()) );
    }

    fn cloned() -> Self {
        Self::instance().with(|s| s.borrow().as_ref().unwrap().clone() )
    }

    fn take() -> Unique<Self> {
        Self::instance().with(|s| Unique(s.borrow_mut().take().unwrap()))
    }

    fn try_take() -> Option<Unique<Self>> {
        Self::instance().with(|s| s.borrow_mut().take().map(|s| Unique(s) ))
    }

    // This panic should never happen in practice when run at the destructor,
    // because if the value is already taken, the panic would happen at take() or try_take(),
    // which makes the object always None when the destructor runs. But if the user calls
    // this explicitly, then the panic might happen.
    fn set(val : Self) {
        Self::instance()
            .with(|s| {
                let mut singleton = s.borrow_mut();
                if singleton.is_none() {
                    *singleton = Some(val)
                } else {
                    panic!("Value already taken")
                }
            })
    }

}

/* Unique is a guard for a Singleton object for which an instance
was acquired via Singleton::take(.). When this guard goes out of scope,
the value is re-written back to the Singleton thread-local storage. While
this guard is alive, the singleton location holds None. As with
other cell-like guards, this implements Deref<Target=T> and DerefMut<Target=T>,
so its contents and methods can be accessed with the dot (.) operator. */
pub struct Unique<T>(T)
where
    T : Singleton;

impl<T> Deref for Unique<T>
where
    T : Singleton
{

    type Target = T;

    fn deref(&self) -> &T {
        &self.0
    }
}

impl<T> DerefMut for Unique<T>
where
    T : Singleton
{

    fn deref_mut(&mut self) -> &mut T {
        &mut self.0
    }

}

impl<T> Drop for Unique<T>
where
    T : Singleton
{

    fn drop(&mut self) {
        T::set(std::mem::take(&mut self.0))
    }

}

// This can be generated by a declarative macro #[derive(Singleton)].
// The module scope guarantees Singleton is only accessible via the trait.
mod s_singleton {

    use super::Singleton;
    use std::cell::RefCell;
    use std::thread::LocalKey;
    use super::S;

    thread_local!(static S_SINGLETON : RefCell<Option<S>> = RefCell::new(None));

    impl Singleton for S {
        fn instance() -> &'static LocalKey<RefCell<Option<S>>> {
            &S_SINGLETON
        }
    }
}

use std::cell::{Cell, /*UnsafeCell*/};

/* A sharing smart pointer that implements panic-free interior mutability for late-bound variables.
This type works like Rc<RefCell<T>>, but imposes a few extra restrictions:

- New instances of the shared value can only be acquired from the Owned<T>::share and Owned<T>::share_mut
methods, never from Shared<T> and SharedMut<T>;

- The shared value cannot be accessed through the Owned<T> directly, but only through Shared<T>
(created via the Owned::share) method or SharedMut<T> (created via the Owned::share_mut);

- The shared value can be access only at a "late-bound" stage (meaning after its corresponding share guard
has been dropped);

- There can be only one mutable shared instance, acquired via Owned::share_mut.

While those rules guarantee there will be no aliased mutable references at the late-bound stage
(and therefore no panics for accessing aliased mutable references), panics might happen if
you try to access the values at the "bound" stage (if there are any share guards alive).

Panic-free interior mutability is be implemented by:

1. Every time a value is to be moved into a closure where it will be accessed in the future,
a corresponding Late<T> guard is created. The access to Shareed<T> can happen only after the Late<T>
drops. Explicitly calling std::mem::drop on Late<T> might cause UB.

2. A mutable instance can be created at most once, which is why
share_mut consumes the sharing wrapper Owned<T>. The mutable instance cannot be acquired if
any Late<T> guards are still alive, guaranteeing it is the only owner. Panics happen
at the sharing state, not at the use state, which means that when Owning<T> goes out of
scope, panics should never happen.

While the restriction of having only a single mutable instance can be annoying to deal
with, it is the price we pay for panic-free access, which holds a clear parallel with
the compile-time rules of &mut T, unlike RefCell<T>, which allows the mutable reference to be
acquired at any time (at the cost of runtime panics). If mutation might happen at multiple sites,
consider using channels and message passing to a receiver closure holding the value. */
/*pub struct Shared<T> {

    val : Rc<UnsafeCell<T>>,

    can_access : Rc<Cell<bool>>

}

const SHARE_ERR : &'static str = "Tried to access shared value, but share guard hasn't been dropped yet";

impl<T> Shared<T> {

    pub fn get(&self) -> &T {
        if self.can_access.get() {
            unsafe { &*self.val.get() }
        } else {
            panic!("{}", SHARE_ERR);
        }
    }
}

pub struct SharedMut<T> {

    val : Rc<UnsafeCell<T>>,

    can_access : Rc<Cell<bool>>

}

impl<T> SharedMut<T> {

    pub fn get_mut(&self) -> &mut T {
        if self.can_access.get() {
            unsafe { &mut *self.val.get() }
        } else {
            panic!("{}", SHARE_ERR);
        }
    }

}

pub struct Owned<T> {

    val : Rc<UnsafeCell<T>>,

    guards : Cell<usize>,

    can_access : Rc<Cell<bool>>

}

impl<T> Owned<T> {

    pub fn new(val : T) -> Self {
        Self { val : Rc::new(UnsafeCell::new(val)), guards : Cell::new(0), can_access : Rc::new(Cell::new(false)) }
    }

    pub fn share<F>(&self, f : F)
    where
        F : Fn(&Shared<T>)
    {

    }

    pub fn share_mut<F>(&self, F : F)
    where
        F : Fn(&mut SharedMut<T>)
    {

    }

    /*pub fn share(&self) -> (Shared<T>, Late) {
        let can_access = self.can_access.clone();
        let shared = Shared { val : self.val.clone(), can_access : can_access.clone() };
        self.guards.set(self.guards.get() + 1);
        (shared, Late { can_access, guards : &self.guards })
    }

    pub fn try_share_mut(self) -> Result<SharedMut<T>, Self> {
        if self.guards.get() == 0 {
            Ok(self.share_mut())
        } else {
            Err(self)
        }
    }

    pub fn share_mut(&self) -> SharedMut<T> {
        if self.guards.get() == 0 {
            SharedMut { val : self.val.clone(), can_access : self.can_access.clone() }
        } else {
            panic!("Cannot acquired mutable shared instance ({} share guards still alive)", self.guards.get());
        }
    }*/

}

impl<T> Drop for Owned<T> {

    fn drop(&mut self) {
        // assert!(self.guards.get() == 0 );
        self.can_access.set(true);
    }

}

pub struct Late<'a> {
    can_access : Rc<Cell<bool>>,
    guards : &'a Cell<usize>
}

impl<'a> Drop for Late<'a> {

    fn drop(&mut self) {
        // self.can_access.set(true);
        self.guards.set(self.guards.get() - 1);
    }

}

// cargo test -- share --nocapture
#[test]
fn share() {

    use std::panic::catch_unwind;

    pub struct MyType(i32);

    fn simple() {
        let f1 = {
            let v = Owned::new(MyType(0));
            let (vs, _guard) = v.share();

            // Accessing vs here would panic, since owned is still alive.
            // v.share_mut() here would panic, since at least one share guard is still alive.
            // share_mut requires all share guards to be zero.

            move || {
                println!("{}", vs.get().0);
            }
            // Calling f1 here would panic, since owned is still alive.
        };

        // f1 can be safely called here.
        f1();
    }

    fn tricky() {
        let mut v = Owned::new(MyType(0));
        // Lets drop the guard and keep a value alive.
        let v2 = {
            let (vs, guard) = v.share();
            vs
        };
        let vm = v.share_mut();
        std::mem::drop(&mut v);
        println!("{}", vm.get_mut().0);
        // println!("{}", v2.get().0);
    }

    tricky();

}*/

/** Owning is a promise from the part of the user that this
type is composed exclusively of types that are either plain-old data or single-ownership pointers
such as Vec<T> and Box<T> where T : Owning, without any shared-ownership
pointers such as Rc<T> or Arc<T>. */
pub unsafe trait Owning {

}

/*unsafe impl<T> Owning for std::rc::Rc<T> {

    fn assert_owning(&self) {
        panic!("Invalid owning type");
    }

}

unsafe impl<T> Owning for std::sync::Arc<T> {

    fn assert_owning(&self) {
        panic!("Invalid owning type");
    }

}*/

/*
/** Wrapper that implements the deferred mutability pattern. This pattern
is a panic-free alternative to guard-based runtime exclusive ownership verification
(i.e. Rc<RefCell<T>>). Instead of acquiring mutable references to a memory location,
updates are made by queing an "update" closure via the Owned::update method, that is lazily
evaluated before any reads of Shared<T>. When the value is to be read, the queue executes
their code if the value is stale, yielding the most recent state, and just yields the old value if no
new updates have been queued since the last read. This gives an API that mimics shared, mutable
memory locations, but each Shared<T> instance keeps its own value, really, sharing
only the update operation across instances. Since all access to the interior mutable location is done inside the
implementation, there is no risk of reading from aliased mutable references at the API level (and
causing runtime panics). There is of course an extra runtime penalty (since the update
code must be executed N times for N separate first-time reads after every update),
so for expensive operations and/or values that will be read from many places  you might still
prefer Rc<RefCell<T>>. But if your updates are cheap, and/or must be read from just a few locations,
Shared<T> might be worth the extra overhead, since you know for sure there won't be any runtime panics.
Since all reads happen behind a closure view(), the only way to access a mutable memory address during
a view would be if the Shared<T> or Owned<T> is moved into the closure, which never ordinarily happens
for &Owned<T> and &Shared<T> because the closure borrows Owned<T>/Shared<T> by reference,
and neither Owned<T> nor Shared<T> are clonable.
The user could of course move Owned<T> into a closure passed into Shared<T>::view and call update(.)
there: But this doesn't violate exclusive ownership guarantees, since the inner values of Owned<T>
and Shared<T> are separate values, and is precisely the reason the deferred mutability pattern is required
(the situation would simply be confusing since the viewed value is one, and the updated value is another,
but doesn't violate the compiler's guarantees). So in essence:

- Moving Owned<T> into Owned<T>::update would be the only way to alias a mutable reference with another mutable
reference, but this isn't allowed by the compiler, since Owned<T>::update takes Owned<T> by reference (&self),
and Owned<T> isn't clonable.

- Moving Owned<T> into Shared<T>::view allows one to call update inside view, but this is a no-issue since each one of Owned<T>
and Shared<T> keep a separate copy of the inner value: the shared value taken as the closure argument is an "old"
value before the update, and the update called inside Shared::view only affects the value at the next call to view.
This is confusing, but doesn't violate compiler guarantees. This single case is the only reason why deferred mutability is required,
instead of just sharing the value itself.

While we may have many Shared<T> instances, we must have a single Owned<T> instance, since otherwise
the order of updates would be very hard to keep track of. Also, if we could move one Owned<T> into another
Owned<T>, we could call updates recursively. To protect against UB in those situations, a panic actually happens
here, since the update op is protected by a RefCell. So while the no-panic guarantee exists for &Owned<T>, it
does not exist for Rc<Owned<T>> by necessity. The reason we have a single Owned<T> is why this structure is panic-free.
If the user attempts to "cheat" by passing Rc<Owned<T>> into Rc<Owned<T>>::update, the second time the
update or view methods try to borrow the op, we would have a panic.

Notice this wrapper is meant to wrap plain-old data. It relies on the deep-clone semantics of the
underlying structure in its share(.) method. Wrapping shared-ownership pointers such as
Rc<RefCell<T>> or Arc<Mutex<T>> means forgoing all no-panic guarantees (think recursively calling lock() or borrow_mut()
inside nested calls to view) and even worse might lead to UB a a few special cases
(that shouldn't really appear in practice but are possible nevertheless). In the case two above
(calling Owned::<T>::update inside Shared<T>::view) might mean having a mutable
reference acquired via the Arc::get_mut or Rc::get_mut from the update argument and a second mutable reference acquired
via the borrow_mut() or lock() of the view argument. If there were a generic way to guarantee at compile time T does
not follow single-threaded or multi-threaded shared ownership semantics (similar to !Sync), we could do that
to make this structure 100% statically-checked safe, but since there isn't, we must live with that caveat. We
could impose for example bytemuck::POD on T, but that is overly restrictive, since we also safely wrap
single-ownership pointers such has Vec<T> or Box<T>.

Notice another way to cause UB is to move Owned<T> into Rc<T> or Arc<T>, therefore making
it clonable/sharable with its previous instance update(.) method. While we can naturally prevent
wrappinig Shared<T> into being wrapped into Arc<T> since it is !Sync, there is no way to guarantee at compile
time Shared<T> cannot be wrapped into a Rc<T>. So we must have a way to deal with recursive calls to update,
since the user could theoretically wrap Owned<T> into Rc<T> or similar and pass it to update of a previous instance.
So while Shared<T> is panic-free, we should make Rc<Owned<T>> panic if there are any recursive calls to update.

While those checks are not provided by the compiler, we trust the user to implement the Owning trait
(which is unsafe to account for those special cases).

Summary:

(1) Recursive calls to &Shared::view across different instances: Do not violate guarantees.

(2) Recursive calls to &Owned::view across different instances: Do not violate guarantees.

(3) Recursive calls to Rc<Owned>::update, or to Rc::Owned::update from Rc::Owned::view, or vice-versa: Checked at runtime
(no-panic guarantee is only for &Owned, not Rc<Owned>).

(4) Calls to Owned::update from Shared::view or to Shared::view from Owned::update: will never panic, since those
types hold separate instances kept synchronized separately. This is where the deferred mutability pattern
kicks in to avoid runtime panics. This is the most common situation in practice, since the user is not
expected to wrap Owned<T> in Rc.

The cost of updating can be amortized by having shared values created from other shared values share
their inner value (second-hand sharing): The first value that updates updates the whole sharing DAG after this point, so
instead of N reads that must be kept synchronized, we have at most 2 reads from a chain that shares
from a single shared value. */
pub struct Shared<T>
where
    T : Owning
{

    // The actual value, shared between the first-hand shared value
    // and any second-hand shared values. A mutable borrow is acquired
    // only at self.try_update.
    val : Rc<UnsafeCell<T>>,

    // The actual updating operation (if any). Called before any
    // reads if the object is "stale" (i.e. if there was a call to
    // Owned::update before this read).
    op : Rc<RefCell<Option<Box<dyn Fn(&mut T)>>>>,

    // If the last update op was applied to this specific instance
    // or to any second-hand shared instances. The first second-hand
    // shared instance, or the original one, that verifies stale==true
    // on a view call is responsible for updating the object.
    stale : Rc<Cell<bool>>,

}

impl<T> Shared<T>
where
    T : Owning
{

    fn try_update(&self) {
        if self.stale.get() {

            // This is the heart of the deferred-mutability pattern: This
            // panic should never happen, even if this Shared<T> is passed into
            // the Owned::<T>::update call, because the op is always borrowed immutably
            // (except at update, where it must be mutably borrowed). If this happens to be
            // called from inside update, just ignore it. The invariant that must be held at the
            // API level is: updates are visible only after the call to update leaves, not inside them.
            // Reading from shared values inside the update method just yields the old value, but does not panic.
            if let Ok(opt_op) = self.op.try_borrow() {
                if let Some(op) = &*opt_op {
                    unsafe { op(&mut *self.val.get()) };
                    self.stale.set(false);
                }
            }
        }
    }

    // Notice sharing values from shared instances itself means sharing operations
    // do not need to be syncrhonized (only shared values shared directly from owned)
    // needs to keep their values syncronized). The first value in the chain that
    // updates makes updates visible for the whole shared chain. This "second-hand sharing"
    // reduces the cost of N update operations for N instances shared from Owned<T> to a single
    // update operation on the first shared instance that notices the value is stale. The change
    // is automatically propagated to the whole rest of the chain.
    pub fn share(&self) -> Self {
        Self {
            val : self.val.clone(),
            op : self.op.clone(),
            stale : self.stale.clone()
        }
    }

    pub fn view<F>(&self, f : F)
    where
        F : Fn(&T) + 'static
    {
        self.try_update();
        unsafe { f(&*self.val.get()); }
    }

}

// Deref is unsound
/*impl<T> Deref for Shared<T> {

    type Target = T;

    fn deref(&self) -> &T {
        self.try_update();
        unsafe { &*self.val.get() }
    }

}*/

/* Note the value cannot be read directly from Owned<T> unless from
the instance inside the update(.) call. */
pub struct Owned<T>
where
    T : Clone + Owning
{

    // Keeps shared stale value will all Shared<T> instances,
    // to notify them of any pending updates.
    stale : RefCell<Vec<Rc<Cell<bool>>>>,

    // Note we use a RefCell here to avoid the risk of recursive calls to update
    // (Only possible if the user wraps Owned<T> into Rc<T>, not possible otherwise).
    val : UnsafeCell<T>,

    op : Rc<RefCell<Option<Box<dyn Fn(&mut T)>>>>,

}

impl<T> Owned<T>
where
    T : Clone + Owning
{

    pub fn new(val : T) -> Self {
        Self {
            val : UnsafeCell::new(val),
            op : Default::default(),
            stale : Default::default()
        }
    }

    pub fn share(&self) -> Shared<T> {
        let stale = Rc::new(Cell::new(true));
        self.stale.borrow_mut().push(stale.clone());
        Shared {
            val : unsafe { Rc::new(UnsafeCell::new((&*self.val.get()).clone())) },
            op : self.op.clone(),
            stale
        }
    }

    pub fn view<F>(&self, f : F)
    where
        F : Fn(&T) + 'static
    {
        // Guard against nested calls of view inside update calls, again, only possible if the
        // user wraps Owned<T> in an Rc.
        assert!(self.op.borrow().is_some());

        // Soundness: the previous assert guarantees this isn't being
        // called inside the scope introduced by Self::update.
        unsafe { f(&*self.val.get()) };
    }

    pub fn update<F>(&self, f : F)
    where
        F : Fn(&mut T) + 'static
    {
        // Note retrieving the op before any access to the inner value is critical, to avoid recursive
        // calls to update (possible if the user wraps Owned<T> in a Rc).
        let mut opt_op = self.op.borrow_mut();

        // Soundness: another reference to value can only live on the same scope
        // on the case of recursive calls to update. Recursive calls should have
        // panicked by now when opt_op is borrowed.
        unsafe { f(&mut *self.val.get()) };

        *opt_op = Some(Box::new(f));
        self.invalidate();
    }

    /** Requires that the queued mutate operation (if any) is executed before the
    next read for this and all shared instances. This saves the re-allocation
    performed by the update(.) call, when the update operation isn't dependent on some
    state at the mutating instance (think an increment operation that does not change
    depending on the current state.) */
    pub fn invalidate(&self) {
        self.stale.borrow_mut().iter().for_each(|s| s.set(true) );
    }

}

// Deref is unsound
/*impl<T> Deref for Owned<T>
where
    T : Clone
{

    type Target = T;

    fn deref(&self) -> &T {
        unsafe { &*self.val.get() }
    }

}*/

#[test]
fn shared() {

    // Note: this is unsound, since the first borrow will have
    // a different value than the second borrow (if we were to
    // keep the first borrow around and compare with the second borrow).
    let val = Owned::new(1usize);
    let s1 = val.share();
    s1.view(|s1| println!("{}", *s1) );
    val.update(|s| *s = 2 );
    s1.view(|s1| println!("{}", *s1) );
}

/*

pub struct OwnedMut<T> {

}

// First-hand, second-hand, etc.
pub struct View<T> {

}

// Notice how this avoids panics: Since Owned<T> is by &mut (e.g. in glib::attach closure
// that works via an FnMut), it is guaranteed to be the only shared place where borrow_mut() is
// called (wrapped in an API).

pub struct Owned<T> {
    val : Rc<RefCell<Option<T>>>
}

impl Owned<T> {

    pub fn update(&mut self, op : F)
    where
        F : FnMut(&mut T)
    {
        op(&mut *self.val.borrow_mut());
    }

}

*/

*/

/*// Implements the exclusive mutability pattern. This type minimizes (but does not
// guarantee the absence of) exclusive mutability panics at runtime.
// This is done by prohibititng recursive calls to update(.) By having Owned<T> be
// an exclusive object (no clone/share impls), and only allowing updates from inside
// a 'static closure that cannot capture the instance itself.
// There is no no-panic guarantees for Shared::view, since calling Shared::view
// from within an update call will panic. There also are no no-panic guarantees
// for Shared::view if a owned instance is moved to the view method, and Owned::update
// is called there. In essence, the extra protection applies only as long as there
// are no nested view-on-update and update-on-view calls. (Recursive views are allowed;
// recursive updates are impossible).
// The no-panic guarantees for the update method not extend to Rc<Owned<T>>,
// since shared instances can be freely passed to it.
pub struct Owned<T>(Rc<RefCell<T>>);

impl<T> Owned<T> {

    pub fn new(val : T) -> Self {
        Self(Rc::new(RefCell::new(val)))
    }

    pub fn share(&self) -> Shared<T> {
        Shared(self.0.clone())
    }

    pub fn view<F,R>(&self, f : F)->R
    where
        F : Fn(&T)->R + 'static
    {
        f(&*self.0.borrow())
    }

    // update can panic if user calls Shared<T>::view
    // from inside F. Which is why the update(.) method
    // is private, and called from behind the reactive
    // mutability trait.
    pub fn update<F, R>(&self, f : F)->R
    where
        F : FnOnce(&mut T)->R + 'static
    {
        f(&mut *self.0.borrow_mut())
    }

}

/* A basic wrapper for Rc<RefCell<T>>, shared by no-panic
runtime exclusive reference checks. */
pub struct Shared<T>(Rc<RefCell<T>>);

impl<T> Shared<T> {

    pub fn share(&self) -> Shared<T> {
        Shared(self.0.clone())
    }

    pub fn view<F,R>(&self, f : F)->R
    where
        F : Fn(&T)->R + 'static
    {
        f(&*self.0.borrow())
    }

}

// Defered mutability objects abstract mutation via a LazyOwned implementor, that applies the
// update on behalf of the object.
// The deferred mutability pattern effectively implements the no-panic guarantee
// for both LazyOwned::update, LazyShared::view and LazyOwned::view. This restricts all mutation
// on the inner T to the Reactive::update method. Since this is a plain function,
// users cannot violate the guarantees by capturing a Shared<T> inside Owned::update,
// as long as message is a "panic-safe" type such a data-carrying enum or a fn pointer.
// If the message can capture its context, then the user can break guarantees by making
// the message for example a closure Box<dyn &mut Self> that captures another shared instance.
pub trait Reactive {

    type Message;

    fn update(&mut self, msg : Self::Message);

}

// Implements the reactive mutability pattern. This restricts all mutability
// to the inner T to a function that receives a message, therefore having
// no way of Owned::update to capture Shared<T>, thus giving a no-panic guarantee.
// This is effectively an wrapper for Owned<T>, that forbids capture at the
// update(.) method.
pub struct ReactOwned<T, M>
where
    T : Reactive<Message=M>
{
    owned : Owned<T>
}

impl<T, M> ReactOwned<T, M>
where
    T : Reactive<Message=M>,
    M : 'static
{

    pub fn share(&self) -> Shared<T> {
        self.owned.share()
    }

    pub fn view<F,R>(&self, f : F)->R
    where
        F : Fn(&T)->R + 'static
    {
        self.owned.view(f)
    }

    pub fn new(val : T) -> Self {
        Self { owned : Owned::new(val) }
    }

    pub fn update(&mut self, msg : M) {
        self.owned.update(|owned| Reactive::update(owned, msg) );
    }

}

// Implements the deferred mutability pattern. This protects against
// recursive calls to update(.) by applying the mutating operation lazily
// before the next view.
pub struct LazyOwned<T> {
    val : Rc<RefCell<T>>,
    op : Rc<RefCell<Option<Box<dyn FnOnce(&mut T) + 'static>>>>
}

impl<T> LazyOwned<T> {

    pub fn share(&self) -> Defered<T> {
        Defered { val : self.val.clone(), op : self.op.clone() }
    }

    // Since op is never called in the update(.) method itself, there
    // is no risk of recursive calls. The point of the deferred mutability
    // pattern is to have an exclusive Rc<RefCell<T>> to which the operation can
    // be written, an many shared values that can only apply this operation, and
    // never write to it. Update cannot be called recursively because 'static.
    // But Defered::view
    pub fn update<F>(&self, op : F)
    where
        F : FnOnce(&mut T) + 'static
    {
        // Must call op here to keep state consistent.
        // cannot panic because DeferOwned is exclusive.
        let mut old_op = self.op.borrow_mut();
        if let Some(old_op) = old_op.take() {
            old_op(&mut *self.val.borrow_mut());
        }
        *old_op = Some(Box::new(op));
    }

}

/* Defered<T> inherits all no-panic guarantees of Owned<T> (no panics at update
since change happens in a 'static closure), no panics at recursive view since value is
immutably borrowed. But it has one extra guarantee: No panics at Defered::view nested
on DeferedOwned::update, because the update method does not borrow the value mutably.
But you might see a stale value on nested calls to view inside DeferOwned::update,
because the guarantee is that the value is the most recent up to the last completion of the update closure.
If the view cannot borrow the op mutably, the old value is shown. Panics never happen on
deferred::update nested inside deferred::view.

The main point of the deferred mutability pattern is that the update(.) method do not
need to take a mutable reference to value **unless** there is a pending update. Inside view calls,
there will never be a pending update, so it just queues the op without taking a mutable reference to value,
so you are free to call update() inside view() without panic.

You are also always free to call view(.) inside update(.), because update(.) never calls its closure
eagerly. It only just queues it before the next retrieval. */
pub struct LazyShared<T> {

    val : Rc<RefCell<T>>,

    op : Rc<RefCell<Option<Box<dyn FnOnce(&mut T) + 'static>>>>

}

impl<T> LazyShared<T> {

    pub fn share(&self) -> LazyShared<T> {
        LazyShared { val : self.val.clone(), op : self.op.clone() }
    }
}*/

/* The panic-free interior mutability patterns wrap standard library's Rc<RefCell<T>>.
Instead of depending on the user keeping track of the borrow guards, all access to
the inner value is done through view(.) and update(.) closures, that keep track
of those guards internally, offering a panic-free API. All patterns are designed around
a single "owning" instance (that can be mutated), and many immutable instances.

All view/update methods have a 'static bound, that makes impossible for the same instance
to call view/update inside its own view/update clousure, since &self cannot be captured inside the
closure. Since the Owned_ variants are exclusive, the only way to to that is to wrap Owned<T> in an
Rc, but the no panic guarantees do not apply in this case (we assume you have a single Owned<T>).

We exclude the possibility of panics by describing what happens when we have recursive
view/update calls or nested view-on-update or update-on-view for shared instances. We also
present what those methods implemented on plain Rc/RefCell would look like without bounds restrictions
for comparison.

-- No-panic guarantee table --

| Pattern              | Update nested on view | View nested on update | Recursive view | Recursive update |
|----------------------|-----------------------|-----------------------|----------------|------------------|
| Plain Rc<RefCell<T>> | Panic                 | Panic                 | Innocuous      | Panic            |
| Exclusive mutability | Panic                 | Panic                 | Innocuous      | Impossible       |
| Defered mutability   | Lazy                  | Stale                 | Innocuous      | Impossible       |
| Reactive mutability  | No capture            | No capture            | Innocuous      | Impossible       |

Innocuous: (only immutable borrow of value happens)
No capture: Mutable borrow happens at implementation, that guarantees no captures of other instances happen.
Impossible: Forbidden by compiler ('static bound for exclusive value)
Lazy: Operation is lazily evaluated, so there is no worries about borrowing mutably twice.
Stale: If op cannot be borrowed, old value is shown instead.

-- Runtime cost table --

| Pattern              | Cost                                             |
|----------------------|--------------------------------------------------|
| Exclusive mutability | Same as Rc<RefCell<T>>                           |
| Defered mutability   | One extra heap allocation at every update        |
| Reactive mutability  | Destructuring a tuple message before each update |

Exclusive mutability is a plain Rc<RefCell<T>>, the cheapest solution, but does not guarantee
no-panics if there are views nested inside updates.
*/

/*pub struct Owned<T> {
    // op : Rc<RefCell<Option<Box<dyn FnOnce(&mut T) + 'static>>>>,
    val : Rc<RefCell<T>>
}

impl<T> Owned<T> {

    // pub fn borrow_mut(&mut self)

    pub fn update<F>(&self, mut op : F)
    where
        F : FnMut(&mut T) + 'static
    {
        // Must call op here to keep state consistent.
        // cannot panic because DeferOwned is exclusive.
        /*let mut old_op = self.op.borrow_mut();
        if let Some(old_op) = old_op.take() {
            old_op(&mut *self.val.borrow_mut());
        }
        *old_op = Some(Box::new(op));*/
        op(&mut *self.val.borrow_mut());
    }

}

// Implements the deferred mutability pattern (allowing for panic-free interior mutability).
// By having a single Owned<T> that is not accessible outside of its creation closure, Shared<T>
// can be freely accessed.
pub struct Shared<T> {
    // op : Rc<RefCell<Option<Box<dyn FnOnce(&mut T) + 'static>>>>,
    val : Rc<RefCell<T>>
}

impl<T> Shared<T> {

    pub fn share(&self) -> Self {
        Self { val : self.val.clone() }
    }

    pub fn view(&self)->std::cell::Ref<T>
    // where
    //    F : Fn(&T)->R + 'static
    {
        // Here is the center of the pattern: val is only accessed immutably
        // after the borrow mut dies, in a controlled manner, inside this implementation
        // (and Shared::view can never be called inside Owned::update, since the user
        // only acquires the shared instance after the update closure is bound there).
        // the mutable borrow is only taken for the first update when the value is stale.
        // The borrow guard never leaves the scope of this implementation, so the next
        // call of view(.) can assume there is no borrow guard alive.
        // if let Some(op) = self.op.borrow_mut().take() {
        //    op(&mut *self.val.borrow_mut());
        // }
        self.val.borrow()
    }

    // Binding the owned instance here forbids any nested calls of update inside view
    // or view inside update.
    pub fn new<F>(val : T, update : F) -> Self
    where
        F : Fn(Owned<T>)
    {
        let val = Rc::new(RefCell::new(val));
        // let op = Rc::new(RefCell::new(None));
        let owned = Owned { /*op : op.clone(),*/ val : val.clone() };
        update(owned);
        Self { val, /*op*/ }
    }

}

//*/ If there is not a channel-like mechanism to use in the update(.)
// call, the LazyOwned type bound via Shared::new_lazy can be used instead: In this case, the
// value itself will queue an op that is applied before any next view(.) call.*/

enum Deferred<T> {
    View(Box<dyn FnMut(&T) + 'static>),
    Update(Box<dyn FnMut(&mut T) + 'static>)
}

// Version using closures to guarantee deferred execution
/// Implements the deferred mutability pattern. This pattern is a panic-free
/// way to guarantee a series of read/writes operations to a shared mutable object
/// is performed in order (although not necessarily in an eager way). In this pattern,
/// read operations are done via the read closure; write operations are done via the write
/// closure. If the operation cannot be performed because the object is borrowed
/// (either a write when it is borrowed for read/write or a read when it is borrowed for write),
/// the operation is queued to be performed at a later stage (when the top-most borrow
/// in the call stack conflicting with the operation drops). This guarantees that no panics
/// occurs and operations are always carried out. Non-nested operations always execute eagerly.
/// If operations are nested, the full op queue is always fully exhausted, as long as the first
/// operation do not conflict with the current view/update.
pub struct Shared<T> {
    val : Rc<RefCell<T>>,
    queue : Rc<RefCell<Vec<Deferred<T>>>>
}

impl<T> Shared<T> {

    pub fn share(&self) -> Self {
        Self { val : self.val.clone(), queue : self.queue.clone() }
    }

    pub fn new(val : T) -> Self {
        Self {
            val : Rc::new(RefCell::new(val)),
            queue : Rc::new(RefCell::new(Vec::new()))
        }
    }

    fn try_process_queue(&self) {
        let mut queue = self.queue.borrow_mut();
        let mut n_processed = 0;
        for op in queue.iter_mut() {
            match op {
                Deferred::View(f) => {
                    if let Ok(val) = self.val.try_borrow() {
                        f(&*val);
                        n_processed += 1;
                    } else {
                        break;
                    }
                },
                Deferred::Update(f) => {
                    if let Ok(mut val) = self.val.try_borrow_mut() {
                        f(&mut *val);
                        n_processed += 1;
                    } else {
                        break;
                    }
                }
            }
        }
        if n_processed >= 1 {
            let mut remaining = queue.split_off(n_processed);
            std::mem::swap(&mut *queue, &mut remaining);
        }
    }

    pub fn try_borrow(&self) -> Option<std::cell::Ref<T>> {
        self.val.try_borrow().ok()
    }

    pub fn try_borrow_mut(&self) -> Option<std::cell::RefMut<T>> {
        self.val.try_borrow_mut().ok()
    }

    pub fn view<F>(&self, mut f : F)
    where
        F : FnMut(&T) + 'static
    {
        if let Ok(val) = self.val.try_borrow() {
            f(&*val);
            self.try_process_queue();
        } else {
            self.queue.borrow_mut().push(Deferred::View(Box::new(f)));
        }
    }

    pub fn update<F>(&self, mut f : F)
    where
        F : FnMut(&mut T) + 'static
    {
        if let Ok(mut val) = self.val.try_borrow_mut() {
            f(&mut *val);
            self.try_process_queue();
        } else {
            self.queue.borrow_mut().push(Deferred::Update(Box::new(f)));
        }
    }

}

fn test() {

    let d : Vec<i32> = Vec::new();
    let a = move || {
        let b = move || {
            d.get(0);
        };
    };
}*/

/* Exclusive types do not contain shared lightweight pointers such as
Rc<T> and Arc<T>. No closure is exclusive, since they can potentially capture
shared pointers. Function pointers, however, can be exclusive. */
pub trait Exclusive {

}

impl Exclusive for () { }

impl Exclusive for bool { }

impl Exclusive for char{ }

impl Exclusive for f32{ }

impl Exclusive for f64{ }

impl Exclusive for i128{ }

impl Exclusive for i16{ }

impl Exclusive for i32{ }

impl Exclusive for i64{ }

impl Exclusive for i8{ }

impl Exclusive for isize{ }

impl Exclusive for str { }

impl Exclusive for u128 { }

impl Exclusive for u16 { }

impl Exclusive for u32 { }

impl Exclusive for u64 { }

impl Exclusive for u8 { }

impl Exclusive for usize { }

impl<T> Exclusive for Box<T>
where
    T : Exclusive
{

}

impl<T> Exclusive for Vec<T>
where
    T : Exclusive
{

}

impl<T, U> Exclusive for BTreeMap<T, U>
where
    T : Exclusive,
    U : Exclusive
{

}

impl<T, U> Exclusive for HashMap<T, U>
where
    T : Exclusive,
    U : Exclusive
{

}

impl<R> Exclusive for fn()->R where R : Exclusive { }

impl<A,R> Exclusive for fn(A)->R where A : Exclusive, R : Exclusive { }

impl<A,B,R> Exclusive for fn(A,B)->R where A : Exclusive, B : Exclusive, R : Exclusive { }

impl<A,B,C,R> Exclusive for fn(A,B,C)->R where A : Exclusive, B : Exclusive, C : Exclusive, R : Exclusive { }

impl<A,B,C,D,R> Exclusive for fn(A,B,C,D)->R where A : Exclusive, B : Exclusive, C : Exclusive, D : Exclusive, R : Exclusive { }

pub trait Reactive {

    type Message : Exclusive;

    fn react(&mut self, msg : Self::Message);

}

/** Implements the deferred mutability pattern. This pattern is a panic-free
alternative to direct use of Rc<RefCell<T>> to share mutable state. In this
pattern, changes to the inner shared value are queued until the outer-most borrow
guard goes out of scope. The user does not have explicit control when the object will actually be mutated,
but this is rather at the discretion of the drop implementation of the View(.) and Update(.)
guards, that execute the update code when a mutable borrow of the shared value is determined to
not panic. To allow code execution when the object actually mutates, the on_changed method allows one to execute code after
the mutation happens. The closures see the state of the object immediately after the change.
The view(.) method offers a borrow guard View(.) that implements Deref to the inner value,
so it can be used for reading (acquiring this guard never panics, as long as the user
honors the Exclusive implementation for T::Message). It is important
the Message type never contains Rc<T> or closures that capture Rc<T> or similar,
causing potential reference cycles which undermine those guarantees. The user must explicitly
state there are no reference cyles by implementing the Exclusive/Unique trait for its object. Message is supposed to be
plain old data, enums containing only plain old data, exclusive owning pointers
and data structures (Box, Vec, etc). Implementing this
for closure wrapper types, although possible, is discouraged, as closures might implicitly capture
objects containing reference cycles. You can never rely on the value being
changed eagerly after the update(.) call (although this might happen), so it is
important to queue any desired reads after the update in on_changed calls instead.
It is best to treat the state of the object immediately after update(.) calls as being undefined,
since this state is dependent on how deep in the borrow hierarchy you are. This pattern is most
useful in situations where there is a clear point where the value is guaranteed to have already
being updated, such as in user interfaces: At every iteration of the main loop, where many shared
instances are distributed among a set of callbacks, you can have an arbitrarily complex
set of immutably borrowed View(.) guards, that all drop at the end of the loop, at which
point the last queued update(.) will be called and its corresponding on_changed argument
afterwards. You can rely on any updates queued at the previous iteration to have already been called
after its end, but you cannot rely on updates queued at the current iteration to have already been called. */
#[derive(Clone)]
pub struct Deferred<T>
where
    T : Reactive
{

    // The inner value, that can always be safely read from, and written to via the deferred
    // mutability pattern.
    val : Rc<RefCell<T>>,

    // Holds the update message queue. The update queue is exhausted when the first borrow guard
    // of the current hierarchy goes out of scope.
    msgs : Rc<RefCell<VecDeque<T::Message>>>,

    // Each guard over this shared object (View<T> and Update<T>) might or might not hold
    // a borrow over the deferred operation as well. The order argument determines if a guard generated
    // by a view(.) or update(.) call will hold a borrow over it: If the order is 0 (first borrow
    // in the hierarchy), it will hold a borrow; else it will not. At every guard drop, the order
    // is decremented by 1 until it goes to zero, and new borrow hierarchy can be generated again.
    order : Rc<Cell<usize>>,

    // Holds the callbacks bound via on_changed. Those callbacks are shared with the first
    // guard in the borrow hierarchy.
    changed : Rc<RefCell<Vec<Box<dyn Fn(&T) + 'static>>>>
}

impl<T> std::fmt::Debug for Deferred<T>
where
    T : Debug + Reactive
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> Result<(), std::fmt::Error> {
        write!(f, "{:?}", *self.view())
    }
}

// Borrow wrapper over an update queue and the on_changed callbacks, held by
// the first borrow guard in a hierarchy.
struct DeferredOp<'a, T>
where
    T : Reactive
{
    msgs : &'a Rc<RefCell<VecDeque<T::Message>>>,
    changed : &'a Rc<RefCell<Vec<Box<dyn Fn(&T) + 'static>>>>
}

struct DeferredGuard<'a, T>
where
    T : Reactive
{
    val : &'a Rc<RefCell<T>>,
    op : Option<DeferredOp<'a, T>>,
    order : Rc<Cell<usize>>
}

impl<'a, T> Drop for DeferredGuard<'a, T>
where
    T : Reactive
{
    fn drop(&mut self) {
        self.order.set(self.order.get() - 1);
        let Some(op) = &self.op else { return };
        let mut msgs = op.msgs.take();
        while let Some(msg) = msgs.pop_front() {
            if let Ok(mut val) = self.val.try_borrow_mut() {
                val.react(msg);
            } else {
                panic!("Tried to borrow val mutably");
            }

            // Re-borrowing the value immutably prevents recursive
            // calls to view from inside the on-changed closure to panic.
            // User is free to call update as well, since op.msgs is not borrowed mutably.
            if let Ok(val) = self.val.try_borrow() {
                // Note: Calling on_changed inside the on_changed callback will panic (since op.changed is borrowed)
                for ch in op.changed.borrow().iter() {
                    ch(&*val);
                }
            } else {
                panic!("Tried to borrow val immutably");
            }
        }
    }
}

/// Borrow guard for the deferred mutability pattern. Implements Deref to the inner value.
pub struct View<'a, T>
where
    T : Reactive
{
    // The borrow is wrapped in an option so we can manipulate the drop
    // order at impl Drop for View. This is always Some(.) while the
    // object is alive, and becomes None only inside the drop implementation.
    borrow : Option<std::cell::Ref<'a, T>>,
    _def : DeferredGuard<'a, T>
}

impl<'a, T> Deref for View<'a, T>
where
    T : Reactive
{

    type Target = T;

    fn deref(&self) -> &T {
        &*self.borrow.as_ref().unwrap()
    }

}

impl<'a, T> Drop for View<'a, T>
where
    T : Reactive
{

    fn drop(&mut self) {
        // It is important to explicitly drop the Ref guard before dropping the Defered
        // (to guarantee the order borrow drops -> can borrow mut at impl Drop for Defered).
        std::mem::drop(self.borrow.take().unwrap());

        // Ok, now drop Defered<T> implicitly.
    }

}

/// Update guard for the deferred mutability pattern. This guard
/// simply exists to determine the potential execution of a deferred
/// operation, and does not provide access to the inner value (since
/// the state of the inner value is undetermined after calls to update,
/// which might have executed eagerly or not).
pub struct Update<'a, T>
where
    T : Reactive
{
    _def : DeferredGuard<'a, T>
}

impl<T> Deferred<T>
where
    T : Reactive
{

    /// Shares another instance of this object.
    pub fn share(&self) -> Self {
        Self {
            val : self.val.clone(),
            msgs : self.msgs.clone(),
            changed : self.changed.clone(),
            order : self.order.clone()
        }
    }

    // The on_changed closure must not contain an update(.) call, or else
    // you will recursively queue and call the update without ever terminating the closure.
    // Panics: Panic if called from within on_changed recursively.
    // Note: Calling view() inside on_changed never panics (but the value will be the same
    // as the argument, so it is best to use that).
    // Note: Calling update() inside the on_changed callback is safe, since the msgs is not borrowed
    // mutably anymore. This is essential in UI code, since updates(.) might be re-entrant when the
    // on_changed callback mutates a widget and the signal is emitted from the callback. The user must
    // be cautious not to change the same widget or widgets pairs in a cyclical fashion, or the
    // on_changed closure will be called endlessly.
    pub fn on_changed<F>(&self, f : F)
    where
        F : Fn(&T) + 'static
    {
        self.changed.borrow_mut().push(Box::new(f));
    }

    pub fn update(&self, msg : T::Message) -> Update<T> {
        self.msgs.borrow_mut().push_back(msg);
        let order = self.order.get();
        let op = if order == 0 {
            Some(DeferredOp { msgs : &self.msgs, changed : &self.changed } )
        } else {
            None
        };
        let def = DeferredGuard {
            val : &self.val,
            op,
            order : self.order.clone()
        };

        // The next views in the call stack will not remember the op
        // or the on_changed closures. The counter is decremented at
        // each View drop impl, until a top-level view is acquired again.
        self.order.set(order + 1);

        Update { _def : def }
    }

    pub fn new(val : T) -> Self {
        Self {
            val : Rc::new(RefCell::new(val)),
            msgs : Rc::new(RefCell::new(VecDeque::new())),
            order : Rc::new(Cell::new(0)),
            changed : Rc::new(RefCell::new(Vec::new()))
        }
    }

    pub fn view(&self) -> View<T> {
        // The zeroth order means this is the first borrow in the call stack:
        // This View should remember the last queued update (if any) and the on_changed
        // closures (if any) to execute when it goes out of scope.
        let order = self.order.get();
        let op = if order == 0 {
            Some(DeferredOp { msgs : &self.msgs, changed : &self.changed } )
        } else {
            None
        };
        let view = View {
            borrow : Some(self.val.borrow()),
            _def : DeferredGuard {
                val : &self.val,
                op,
                order : self.order.clone()
            }
        };

        // The next views in the call stack will not remember the op
        // or the on_changed closures. The counter is decremented at
        // each View drop impl, until a top-level view is acquired again.
        self.order.set(order + 1);

        view
    }

}

// cargo test -- deferred --nocapture
#[test]
fn deferred() {

    struct MyStruct(i32);
    struct MyMsg(i32);
    impl Exclusive for MyMsg { }
    impl Reactive for MyStruct {
        type Message = MyMsg;
        fn react(&mut self, msg : MyMsg) {
            self.0 = msg.0;
        }
    }

    let s = Deferred::new(MyStruct(0));
    s.on_changed(|s| { format!("Value now {}", s.0); } );
    for i in 0..3 {
        println!("Iteration {}", i);
        println!("Begin: {} = {}", i, s.view().0);
        s.update(MyMsg(i+1));
        println!("End: {} = {}", i, s.view().0);
    }

}

use std::sync::{Arc, Mutex};

#[derive(Clone)]
pub struct Pool<T> {
    free : Arc<Mutex<Vec<T>>>,
    init : Arc<dyn Fn()->T>
}

pub struct Instance<'a, T> {
    pool : &'a Pool<T>,
    val : Option<T>
}

impl<'a, T> Deref for Instance<'a, T> {

    type Target=T;

    fn deref(&self) -> &T {
        self.val.as_ref().unwrap()
    }
}

impl<'a, T> DerefMut for Instance<'a, T> {

    fn deref_mut(&mut self) -> &mut T {
        self.val.as_mut().unwrap()
    }

}

impl<'a, T> Drop for Instance<'a, T> {

    fn drop(&mut self) {
        let mut free = self.pool.free.lock().unwrap();
        free.push(self.val.take().unwrap());
    }

}

impl<T> Pool<T> {

    pub fn take<'a>(&'a self) -> Instance<'a, T> {
        if let Some(val) = self.free.lock().unwrap().pop() {
            Instance {
                pool : self,
                val : Some(val)
            }
        } else {
            Instance {
                pool : self,
                val : Some((&self.init)())
            }
        }
    }

    pub fn new<F : Fn()->T + 'static>(f : F) -> Self {
        Self::with_capacity(0, f)
    }

    pub fn with_capacity<F : Fn()->T + 'static>(n : usize, f : F) -> Self {
        let mut free = Vec::with_capacity(n);
        for _ in 0..n {
            free.push(f());
        }
        Self {
            free : Arc::new(Mutex::new(free)),
            init : Arc::new(f)
        }
    }
}

/*
The heavy and light types provide wrappers for mutable memory locations
that always move the value. This provides the same API for Heavy<T> and
Light<T>, and never exposes the panicking APIs of RefCell (borrow and borrow_mut).

The take(.) method pretty much follows the semantics of try_borrow(.) (also for
cell types, since it fails if the cell contains the None variant).

The methods with(.) and with_mut(.) gives a clear way to reason about the possibility
of invalid states: An invalid state is only possible if the Light(.) or Heavy(.) is
directly or indirectly captured by the closure passed this function. Sequential calls
to those methods are guaranteed to never fail. Since the closures return an arbitrary
value, anything that might require the shared state recursively directly
(or indirectly via signal/slots mechanisms) can be made after the with call,
potentially using the returned value.

For Heavy<T>, the advantage of having a guard over the value instead of over
the mutable reference is that the mutability of the inner value can be
bound by the user.

Heavy<T> and Light<T> will implement Try (when stabilized) so that the following pattern can be used:

pub struct AppState {

    counter : Light<i32>,

    text : Heavy<String>
}

impl AppState {

    fn text<'a>(self : &'a Rc<Self>) -> Result<impl Deref<Target=String> + 'a, Box<dyn Error>> {
        Ok(self.text?)
    }

    fn increment_counter(self : Rc<Self>) -> Result<(), Box<dyn Error>> {
        self.counter? += 1;
        Ok(())
    }

}

Which is more explicit about AppState always following shared mutability semantics
rather than explicit wrapping Rc<RefCell<T>> and requiring its direct use.
*/

/*
Provides panic-free shared mutable access to a lightweight (copy) type T.
*/
#[derive(Clone, Debug)]
pub struct Light<T>
where
    T : Copy
{
    data : Cell<Option<T>>
}

pub struct A {
    val : Heavy<String>
}

impl A {

    fn val<'a>(self : &'a Rc<Self>) -> impl Deref<Target=String> + 'a {
        self.val.take().unwrap()
    }

}

#[test]
fn test_heavy() {

    let a = Rc::new(A{ val : Heavy::new(String::new()) });
    a.val();

}

impl<T> Light<T>
where
    T : Copy
{

    pub fn new(val : T) -> Self {
        Self { data : Cell::new(Some(val)) }
    }

    pub fn take(&self) -> Option<LightGuard<T>> {
        Some(LightGuard { shared : self, val : Some(self.data.take()?) })
    }

    pub fn with<R>(&self, f : impl Fn(&T)->R) -> Option<R> {
        let obj = self.take()?;
        Some(f(&obj))
    }

    pub fn with_mut<R>(&self, f : impl Fn(&mut T)->R) -> Option<R> {
        let mut obj = self.take()?;
        Some(f(&mut obj))
    }
}

pub struct LightGuard<'a, T>
where
    T : Copy
{
    shared : &'a Light<T>,
    val : Option<T>
}

impl<'a, T> Deref for LightGuard<'a, T>
where
    T : Copy
{

    type Target=T;

    fn deref(&self) -> &T {
        self.val.as_ref().unwrap()
    }
}

impl<'a, T> DerefMut for LightGuard<'a, T>
where
    T : Copy
{

    fn deref_mut(&mut self) -> &mut T {
        self.val.as_mut().unwrap()
    }

}

impl<'a, T> Drop for LightGuard<'a, T>
where
    T : Copy
{

    fn drop(&mut self) {
        let _ = self.shared.data.replace(Some(self.val.take().unwrap()));
    }

}

/*
Provides panic-free shared mutable access to a heavy (non-copy) type T.
*/
#[derive(Clone, Debug)]
pub struct Heavy<T> {
    data : RefCell<Option<T>>
}

pub struct HeavyGuard<'a, T> {
    shared : &'a Heavy<T>,
    val : Option<T>
}

impl<'a, T> Deref for HeavyGuard<'a, T> {

    type Target=T;

    fn deref(&self) -> &T {
        self.val.as_ref().unwrap()
    }
}

impl<'a, T> DerefMut for HeavyGuard<'a, T> {

    fn deref_mut(&mut self) -> &mut T {
        self.val.as_mut().unwrap()
    }

}

impl<'a, T> Drop for HeavyGuard<'a, T> {

    fn drop(&mut self) {
        let _ = self.shared.data.replace(Some(self.val.take().unwrap()));
    }

}

impl<T> Heavy<T> {

    pub fn new(val : T) -> Self {
        Self { data : RefCell::new(Some(val)) }
    }

    pub fn take(&self) -> Option<HeavyGuard<T>> {
        Some(HeavyGuard { shared : self, val : Some(self.data.borrow_mut().take()?) })
    }

    pub fn with<R>(&self, f : impl Fn(&T)->R) -> Option<R> {
        let obj = self.take()?;
        Some(f(&obj))
    }

    pub fn with_mut<R>(&self, f : impl Fn(&mut T)->R) -> Option<R> {
        let mut obj = self.take()?;
        Some(f(&mut obj))
    }

}


